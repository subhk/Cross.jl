# =============================================================================
#  Tri-Global Instability Analysis
#
#  Linear stability analysis for non-axisymmetric basic states where
#  perturbations couple across multiple azimuthal modes m.
#
#  With a basic state containing mode m_bs, the perturbation temperature:
#    θ'(r,θ,φ,t) = Σ_m θ'_m(r,θ,t) e^{imφ}
#  couples modes m and m ± m_bs through advection by the basic state.
#
#  This requires solving a BLOCK-COUPLED eigenvalue problem where different
#  m-blocks are coupled through the basic state.
# =============================================================================

using Parameters
using LinearAlgebra
using SparseArrays
using Printf
using KrylovKit
using WignerSymbols

# Import from parent module
import ..Cross: LinearStabilityOperator, OnsetParams, assemble_matrices

"""
    TriGlobalParams{T<:Real}

Parameters for tri-global instability analysis with non-axisymmetric basic state.

Unlike OnsetParams which solves for a single azimuthal mode m, TriGlobalParams
solves for MULTIPLE coupled modes simultaneously.

Fields:
- `E::T` - Ekman number
- `Pr::T` - Prandtl number
- `Ra::T` - Rayleigh number
- `χ::T` - Radius ratio r_i/r_o
- `m_range::UnitRange{Int}` - Range of perturbation modes to include (e.g., -2:2)
- `lmax::Int` - Maximum spherical harmonic degree
- `Nr::Int` - Number of radial points
- `basic_state_3d::BasicState3D{T}` - The 3D basic state
- `mechanical_bc::Symbol` - :no_slip or :stress_free
- `thermal_bc::Symbol` - :fixed_temperature or :fixed_flux

Note: The size of the eigenvalue problem is ~ length(m_range) × lmax × Nr × 3
which can become very large. Use sparse methods and Krylov subspace solvers.
"""
@with_kw struct TriGlobalParams{T<:Real}
    E::T
    Pr::T
    Ra::T
    χ::T
    m_range::UnitRange{Int}
    lmax::Int
    Nr::Int
    basic_state_3d::BasicState3D{T}
    mechanical_bc::Symbol = :no_slip
    thermal_bc::Symbol = :fixed_temperature
end


"""
    get_coupling_modes(m::Int, m_bs::Int, m_range::UnitRange{Int})

Determine which perturbation modes couple to mode m through basic state mode m_bs.

A basic state with azimuthal mode m_bs couples perturbation modes m and m ± m_bs
through advection terms.

Returns:
- Vector of coupled mode numbers that are within m_range

Example:
    # Basic state with m_bs = 2, perturbation range -3:3
    get_coupling_modes(0, 2, -3:3)  # Returns [-2, 0, 2]
    get_coupling_modes(1, 2, -3:3)  # Returns [-1, 1, 3]
"""
function get_coupling_modes(m::Int, m_bs::Int, m_range::UnitRange{Int})
    coupled_modes = Int[]

    # Mode m couples to m - m_bs, m, m + m_bs
    for Δm in [-m_bs, 0, m_bs]
        m_coupled = m + Δm
        if m_coupled in m_range
            push!(coupled_modes, m_coupled)
        end
    end

    return sort(unique(coupled_modes))
end


"""
    build_mode_coupling_structure(m_range::UnitRange{Int},
                                  basic_state::BasicState3D{T}) where T

Analyze the coupling structure between perturbation modes induced by the basic state.

Returns:
- `coupling_graph::Dict{Int, Vector{Int}}` - For each mode m, which other modes couple to it
- `all_m_bs::Vector{Int}` - All non-zero azimuthal modes in the basic state

This information is used to construct the block-sparse eigenvalue problem.
"""
function build_mode_coupling_structure(m_range::UnitRange{Int},
                                       basic_state::BasicState3D{T}) where T

    # Find all non-zero azimuthal modes in the basic state
    all_m_bs = Int[]
    for ((ℓ, m_bs), theta_coeff) in basic_state.theta_coeffs
        if m_bs != 0 && maximum(abs.(theta_coeff)) > 1e-14
            push!(all_m_bs, m_bs)
        end
    end
    all_m_bs = sort(unique(all_m_bs))

    # Build coupling graph
    coupling_graph = Dict{Int, Vector{Int}}()

    for m in m_range
        coupled_modes = Int[m]  # Always couples to itself

        # Add coupling through each basic state mode
        for m_bs in all_m_bs
            # Basic state mode m_bs couples m to m ± m_bs
            for Δm in [-m_bs, m_bs]
                m_coupled = m + Δm
                if m_coupled in m_range && m_coupled != m
                    push!(coupled_modes, m_coupled)
                end
            end
        end

        coupling_graph[m] = sort(unique(coupled_modes))
    end

    return coupling_graph, all_m_bs
end


"""
    estimate_triglobal_problem_size(params::TriGlobalParams{T}) where T

Estimate the size of the tri-global eigenvalue problem.

Returns:
- `total_dofs::Int` - Total degrees of freedom
- `matrix_size::Int` - Size of the matrix (= total_dofs)
- `num_modes::Int` - Number of coupled azimuthal modes
- `dofs_per_mode::Int` - Degrees of freedom per mode

Useful for assessing computational requirements before attempting to solve.
"""
function estimate_triglobal_problem_size(params::TriGlobalParams{T}) where T
    num_modes = length(params.m_range)
    lmax = params.lmax
    Nr = params.Nr

    # For each m, we have lmax - m + 1 spherical harmonic degrees ℓ ∈ [m, lmax]
    # Each (ℓ,m) has:
    # - Nr coefficients for P_ℓm (poloidal potential)
    # - Nr coefficients for T_ℓm (toroidal potential)
    # - Nr coefficients for Θ_ℓm (temperature)
    # Total: 3 × Nr per (ℓ,m)

    dofs_per_mode = 0
    for m in params.m_range
        num_ell = lmax - abs(m) + 1
        dofs_per_mode += num_ell * 3 * Nr
    end

    total_dofs = dofs_per_mode  # Note: dofs_per_mode already sums over all m
    matrix_size = total_dofs

    return (
        total_dofs = total_dofs,
        matrix_size = matrix_size,
        num_modes = num_modes,
        dofs_per_mode = div(dofs_per_mode, num_modes)  # Average per mode
    )
end


"""
    CoupledModeProblem{T<:Real}

Data structure for the coupled-mode eigenvalue problem.

Fields:
- `params::TriGlobalParams{T}` - Problem parameters
- `m_range::UnitRange{Int}` - Range of coupled modes
- `coupling_graph::Dict{Int,Vector{Int}}` - Mode coupling structure
- `block_indices::Dict{Int,UnitRange{Int}}` - Index ranges for each mode block
- `total_dofs::Int` - Total degrees of freedom

This structure organizes the information needed to assemble and solve the
block-coupled eigenvalue problem:
    A_coupled x = λ B_coupled x
where A_coupled has diagonal blocks (single-mode operators) and off-diagonal
blocks (mode coupling through basic state).
"""
@with_kw struct CoupledModeProblem{T<:Real}
    params::TriGlobalParams{T}
    m_range::UnitRange{Int}
    coupling_graph::Dict{Int,Vector{Int}}
    all_m_bs::Vector{Int}
    block_indices::Dict{Int,UnitRange{Int}}
    total_dofs::Int
end


"""
    setup_coupled_mode_problem(params::TriGlobalParams{T}) where T

Initialize the coupled-mode eigenvalue problem structure.

This analyzes the basic state to determine:
1. Which perturbation modes m couple to each other
2. The index ranges for each m-block in the global matrix
3. The total problem size

Returns a CoupledModeProblem structure.
"""
function setup_coupled_mode_problem(params::TriGlobalParams{T}) where T
    m_range = params.m_range
    basic_state = params.basic_state_3d

    # Analyze coupling structure
    coupling_graph, all_m_bs = build_mode_coupling_structure(m_range, basic_state)

    # Compute index ranges for each mode
    # Each mode m has (lmax - |m| + 1) × 3 × Nr DOFs
    block_indices = Dict{Int,UnitRange{Int}}()
    current_idx = 1

    for m in m_range
        num_ell = params.lmax - abs(m) + 1
        block_size = num_ell * 3 * params.Nr

        block_indices[m] = current_idx:(current_idx + block_size - 1)
        current_idx += block_size
    end

    total_dofs = current_idx - 1

    return CoupledModeProblem(
        params = params,
        m_range = m_range,
        coupling_graph = coupling_graph,
        all_m_bs = all_m_bs,
        block_indices = block_indices,
        total_dofs = total_dofs
    )
end


# =============================================================================
#  Helper Functions for Tri-Global Eigenvalue Problem
# =============================================================================

"""
    build_single_mode_operators(problem::CoupledModeProblem, verbose::Bool)

Build single-mode linear stability operators for each azimuthal mode m.

Returns a dictionary mapping m => (A_m, B_m, op_m) where:
- A_m, B_m are the LHS and RHS matrices for mode m
- op_m is the LinearStabilityOperator for mode m
"""
function build_single_mode_operators(problem::CoupledModeProblem{T}, verbose::Bool) where T
    params_tri = problem.params
    single_mode_ops = Dict{Int, Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}, Any}}()

    for m in problem.m_range
        if verbose && abs(m) <= 2
            print("  m = $m... ")
        end

        # Create OnsetParams for this mode
        params_m = OnsetParams(
            E = params_tri.E,
            Pr = params_tri.Pr,
            Ra = params_tri.Ra,
            χ = params_tri.χ,
            m = abs(m),  # m must be non-negative for OnsetParams
            lmax = params_tri.lmax,
            Nr = params_tri.Nr,
            mechanical_bc = params_tri.mechanical_bc,
            thermal_bc = params_tri.thermal_bc,
            basic_state = nothing  # Don't include basic state in single-mode ops
        )

        # Create operator and assemble matrices
        op_m = LinearStabilityOperator(params_m)
        A_m, B_m, _, _ = assemble_matrices(op_m)

        single_mode_ops[m] = (A_m, B_m, op_m)

        if verbose && abs(m) <= 2
            println("$(size(A_m, 1)) DOFs")
        end
    end

    if verbose && length(problem.m_range) > 5
        println("  ... ($(length(problem.m_range)) modes total)")
    end

    return single_mode_ops
end


"""
    build_mode_coupling_operators(problem::CoupledModeProblem, single_mode_ops::Dict, verbose::Bool)

Build coupling operators between different azimuthal modes through the 3D basic state.

The coupling arises from:
1. **Advection of perturbation by basic state**: (ū_bs · ∇)θ'
   - Basic state flow ū with mode m_bs advects perturbation θ' with mode m_pert
   - Couples m_pert to m_pert ± m_bs through the φ-derivative: ∂/∂φ → im

2. **Perturbation advecting basic state temperature**: (u' · ∇)θ̄_bs
   - Perturbation velocity u' with mode m_pert advects basic state temperature θ̄ with mode m_bs
   - Couples m_pert to m_pert ± m_bs

3. **Shear production**: (u' · ∇)ū_bs
   - Perturbation velocity interacting with basic state velocity gradients

Returns a dictionary mapping (m_from, m_to) => C_{from,to} where C is the
coupling matrix from mode m_from to mode m_to.
"""
function build_mode_coupling_operators end  # Forward declaration

"""
    interpolate_to_grid(coeffs_bs::Vector{T}, r_bs::Vector{T}, r_op::Vector{T}) where T

Interpolate basic state coefficients from the basic state grid to the operator grid.
Uses simple linear interpolation.
"""
function interpolate_to_grid(coeffs_bs::Vector{T}, r_bs::Vector{T}, r_op::Vector{T}) where T
    Nr_op = length(r_op)
    coeffs_interp = zeros(T, Nr_op)

    for i in 1:Nr_op
        r_target = r_op[i]

        # Find bracketing points in r_bs
        if r_target <= r_bs[1]
            coeffs_interp[i] = coeffs_bs[1]
        elseif r_target >= r_bs[end]
            coeffs_interp[i] = coeffs_bs[end]
        else
            # Find j such that r_bs[j] <= r_target < r_bs[j+1]
            j = searchsortedlast(r_bs, r_target)
            if j >= length(r_bs)
                j = length(r_bs) - 1
            end
            # Linear interpolation
            t = (r_target - r_bs[j]) / (r_bs[j+1] - r_bs[j])
            coeffs_interp[i] = (1 - t) * coeffs_bs[j] + t * coeffs_bs[j+1]
        end
    end

    return coeffs_interp
end

function build_mode_coupling_operators(problem::CoupledModeProblem{T},
                                        single_mode_ops::Dict,
                                        verbose::Bool) where T
    coupling_ops = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    params = problem.params
    basic_state = params.basic_state_3d
    Nr = params.Nr
    lmax = params.lmax

    # Get radial grid from one of the single-mode operators
    first_m = first(problem.m_range)
    _, _, op_ref = single_mode_ops[first_m]
    r = op_ref.r
    cd = op_ref.cd

    # Get the basic state radial grid for interpolation
    r_bs = basic_state.r

    # Pre-interpolate basic state coefficients to operator grid if grids differ
    needs_interpolation = (length(r_bs) != length(r)) || (maximum(abs.(r_bs .- r)) > 1e-10)

    # Cache interpolated coefficients
    uphi_interp = Dict{Tuple{Int,Int}, Vector{T}}()
    duphi_dr_interp = Dict{Tuple{Int,Int}, Vector{T}}()
    dtheta_dr_interp = Dict{Tuple{Int,Int}, Vector{T}}()

    for (key, coeff) in basic_state.uphi_coeffs
        if needs_interpolation
            uphi_interp[key] = interpolate_to_grid(coeff, r_bs, r)
        else
            uphi_interp[key] = coeff
        end
    end
    for (key, coeff) in basic_state.duphi_dr_coeffs
        if needs_interpolation
            duphi_dr_interp[key] = interpolate_to_grid(coeff, r_bs, r)
        else
            duphi_dr_interp[key] = coeff
        end
    end
    for (key, coeff) in basic_state.dtheta_dr_coeffs
        if needs_interpolation
            dtheta_dr_interp[key] = interpolate_to_grid(coeff, r_bs, r)
        else
            dtheta_dr_interp[key] = coeff
        end
    end

    # Find all non-zero basic state modes (ℓ_bs, m_bs)
    bs_modes = Tuple{Int,Int}[]
    for ((ℓ_bs, m_bs), coeff) in basic_state.theta_coeffs
        if m_bs != 0 && maximum(abs.(coeff)) > 1e-14
            push!(bs_modes, (ℓ_bs, m_bs))
        end
    end
    # Also check uphi for non-zero modes
    for ((ℓ_bs, m_bs), coeff) in basic_state.uphi_coeffs
        if m_bs != 0 && maximum(abs.(coeff)) > 1e-14
            if (ℓ_bs, m_bs) ∉ bs_modes
                push!(bs_modes, (ℓ_bs, m_bs))
            end
        end
    end

    if verbose
        println("  Non-zero basic state modes: ", bs_modes)
    end

    # For each mode pair that should couple
    n_nonzero_couplings = 0

    for m_from in problem.m_range
        for m_to in problem.m_range
            if m_from == m_to
                continue  # Diagonal is handled separately
            end

            # Check if these modes can couple through any basic state mode
            Δm = m_to - m_from
            can_couple = false
            for (ℓ_bs, m_bs) in bs_modes
                if abs(Δm) == m_bs
                    can_couple = true
                    break
                end
            end

            if !can_couple
                continue
            end

            # Get sizes of the from and to blocks
            _, _, op_from = single_mode_ops[m_from]
            _, _, op_to = single_mode_ops[m_to]
            n_from = length(problem.block_indices[m_from])
            n_to = length(problem.block_indices[m_to])

            # Build the coupling matrix
            C = zeros(ComplexF64, n_to, n_from)

            # Compute coupling through each relevant basic state mode
            for (ℓ_bs, m_bs) in bs_modes
                # Check if this basic state mode couples m_from to m_to
                # The spherical harmonic selection rule requires m_from + m_bs_eff = m_to
                # where m_bs_eff = ±m_bs depending on the coupling direction
                if m_to == m_from + m_bs
                    # Forward coupling: m_from + m_bs = m_to
                    m_bs_eff = m_bs
                    add_advection_coupling!(C, op_from, op_to,
                                           m_from, m_to, ℓ_bs, m_bs_eff,
                                           r, uphi_interp, params)
                    add_temperature_gradient_coupling!(C, op_from, op_to,
                                                       m_from, m_to, ℓ_bs, m_bs_eff,
                                                       r, dtheta_dr_interp, params)
                    add_shear_coupling!(C, op_from, op_to,
                                       m_from, m_to, ℓ_bs, m_bs_eff,
                                       r, duphi_dr_interp, params)
                elseif m_to == m_from - m_bs
                    # Backward coupling: m_from - m_bs = m_to → m_from + (-m_bs) = m_to
                    # Need to use -m_bs in the Gaunt coefficient (complex conjugate of Y_{ℓ,m})
                    m_bs_eff = -m_bs
                    add_advection_coupling!(C, op_from, op_to,
                                           m_from, m_to, ℓ_bs, m_bs_eff,
                                           r, uphi_interp, params)
                    add_temperature_gradient_coupling!(C, op_from, op_to,
                                                       m_from, m_to, ℓ_bs, m_bs_eff,
                                                       r, dtheta_dr_interp, params)
                    add_shear_coupling!(C, op_from, op_to,
                                       m_from, m_to, ℓ_bs, m_bs_eff,
                                       r, duphi_dr_interp, params)
                end
            end

            # Store if non-zero
            if maximum(abs.(C)) > 1e-16
                coupling_ops[(m_from, m_to)] = C
                n_nonzero_couplings += 1
            end
        end
    end

    if verbose
        println("  Built $n_nonzero_couplings non-zero coupling blocks")
    end

    return coupling_ops
end


"""
    add_advection_coupling!(C, op_from, op_to, m_from, m_to, ℓ_bs, m_bs, r, uphi_coeffs, params)

Add advection coupling: (ū_bs · ∇)θ' contribution to the temperature equation.

The azimuthal advection term is:
    (ū_φ,bs / (r sin θ)) ∂θ'/∂φ = im_from × ū_φ,bs/(r sin θ) × θ'

This couples perturbation mode m_from to m_to = m_from ± m_bs through the
product of spherical harmonics.

Arguments:
- uphi_coeffs: Dictionary of interpolated uphi coefficients on the operator grid
"""
function add_advection_coupling!(C::Matrix{ComplexF64},
                                  op_from, op_to,
                                  m_from::Int, m_to::Int,
                                  ℓ_bs::Int, m_bs::Int,
                                  r::Vector{T},
                                  uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
                                  params) where T
    Nr = length(r)
    m_pert_from = abs(m_from)
    m_pert_to = abs(m_to)

    # Get basic state zonal flow coefficient for (ℓ_bs, |m_bs|) - already interpolated
    # Note: m_bs can be negative for backward coupling, but coefficients are stored with positive m
    key_bs = (ℓ_bs, abs(m_bs))
    if !haskey(uphi_coeffs, key_bs)
        return  # No coupling from this mode
    end
    uphi_bs = uphi_coeffs[key_bs]

    if maximum(abs.(uphi_bs)) < 1e-14
        return  # Negligible basic state flow
    end

    # Advection coefficient: im_from × ū_φ/(r)
    # Note: The 1/sinθ factor is handled through spherical harmonic coupling
    adv_coeff = im * m_from .* uphi_bs ./ r

    # Loop over ℓ modes in both from and to operators
    for (ℓ_from, field_from) in keys(op_from.index_map)
        if field_from != :Θ
            continue  # Only temperature is advected
        end
        if ℓ_from < m_pert_from
            continue
        end

        for (ℓ_to, field_to) in keys(op_to.index_map)
            if field_to != :Θ
                continue
            end
            if ℓ_to < m_pert_to
                continue
            end

            # Compute spherical harmonic coupling coefficient
            # Y_{ℓ_to, m_to} from Y_{ℓ_from, m_from} × Y_{ℓ_bs, m_bs}
            coupling_coeff = compute_sh_coupling_coefficient(
                ℓ_from, m_from, ℓ_bs, m_bs, ℓ_to, m_to
            )

            if abs(coupling_coeff) < 1e-14
                continue
            end

            # Get index ranges
            idx_from = op_from.index_map[(ℓ_from, :Θ)]
            idx_to = op_to.index_map[(ℓ_to, :Θ)]

            # Map to global block indices
            # The op indices are local; we need to find offset within the block
            from_offset = first(idx_from) - 1
            to_offset = first(idx_to) - 1

            # Add diagonal coupling (radial operator)
            for i in 1:Nr
                row = to_offset + i
                col = from_offset + i
                if row <= size(C, 1) && col <= size(C, 2)
                    C[row, col] += coupling_coeff * adv_coeff[i]
                end
            end
        end
    end
end


"""
    add_temperature_gradient_coupling!(C, op_from, op_to, m_from, m_to, ℓ_bs, m_bs, r, dtheta_dr_coeffs, params)

Add coupling from perturbation velocity advecting basic state temperature: (u' · ∇)θ̄_bs

This term appears in the temperature equation and couples the poloidal velocity
(which determines u'_r) to the temperature through the radial gradient of θ̄_bs.

Arguments:
- dtheta_dr_coeffs: Dictionary of interpolated dtheta_dr coefficients on the operator grid
"""
function add_temperature_gradient_coupling!(C::Matrix{ComplexF64},
                                             op_from, op_to,
                                             m_from::Int, m_to::Int,
                                             ℓ_bs::Int, m_bs::Int,
                                             r::Vector{T},
                                             dtheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
                                             params) where T
    Nr = length(r)
    m_pert_from = abs(m_from)
    m_pert_to = abs(m_to)

    # Get basic state temperature gradient for (ℓ_bs, |m_bs|) - already interpolated
    # Note: m_bs can be negative for backward coupling, but coefficients are stored with positive m
    key_bs = (ℓ_bs, abs(m_bs))
    if !haskey(dtheta_dr_coeffs, key_bs)
        return
    end
    dtheta_dr_bs = dtheta_dr_coeffs[key_bs]

    if maximum(abs.(dtheta_dr_bs)) < 1e-14
        return
    end

    # The coupling is: u'_r × ∂θ̄_bs/∂r
    # where u'_r = ℓ(ℓ+1)/r² × P (from poloidal potential)

    for (ℓ_from, field_from) in keys(op_from.index_map)
        if field_from != :P
            continue  # Poloidal potential gives radial velocity
        end
        if ℓ_from < m_pert_from
            continue
        end

        L_from = ℓ_from * (ℓ_from + 1)

        for (ℓ_to, field_to) in keys(op_to.index_map)
            if field_to != :Θ
                continue  # This goes into temperature equation
            end
            if ℓ_to < m_pert_to
                continue
            end

            # Compute spherical harmonic coupling
            coupling_coeff = compute_sh_coupling_coefficient(
                ℓ_from, m_from, ℓ_bs, m_bs, ℓ_to, m_to
            )

            if abs(coupling_coeff) < 1e-14
                continue
            end

            # Get index ranges
            idx_from = op_from.index_map[(ℓ_from, :P)]
            idx_to = op_to.index_map[(ℓ_to, :Θ)]

            from_offset = first(idx_from) - 1
            to_offset = first(idx_to) - 1

            # Coupling coefficient: L_from/r² × ∂θ̄_bs/∂r
            temp_grad_coeff = L_from .* dtheta_dr_bs ./ (r.^2)

            for i in 1:Nr
                row = to_offset + i
                col = from_offset + i
                if row <= size(C, 1) && col <= size(C, 2)
                    C[row, col] += coupling_coeff * temp_grad_coeff[i]
                end
            end
        end
    end
end


"""
    add_shear_coupling!(C, op_from, op_to, m_from, m_to, ℓ_bs, m_bs, r, duphi_dr_coeffs, params)

Add shear production coupling: (u' · ∇)ū_bs contribution to momentum equations.

This couples perturbation poloidal velocity to toroidal velocity through
the basic state velocity gradients.

Arguments:
- duphi_dr_coeffs: Dictionary of interpolated duphi_dr coefficients on the operator grid
"""
function add_shear_coupling!(C::Matrix{ComplexF64},
                              op_from, op_to,
                              m_from::Int, m_to::Int,
                              ℓ_bs::Int, m_bs::Int,
                              r::Vector{T},
                              duphi_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
                              params) where T
    Nr = length(r)
    m_pert_from = abs(m_from)
    m_pert_to = abs(m_to)

    # Get basic state velocity gradient - already interpolated
    # Note: m_bs can be negative for backward coupling, but coefficients are stored with positive m
    key_bs = (ℓ_bs, abs(m_bs))
    if !haskey(duphi_dr_coeffs, key_bs)
        return
    end
    duphi_dr_bs = duphi_dr_coeffs[key_bs]

    if maximum(abs.(duphi_dr_bs)) < 1e-14
        return
    end

    # Shear term: u'_r × ∂ū_φ/∂r couples poloidal (P) to toroidal (T)

    for (ℓ_from, field_from) in keys(op_from.index_map)
        if field_from != :P
            continue
        end
        if ℓ_from < m_pert_from
            continue
        end

        L_from = ℓ_from * (ℓ_from + 1)

        for (ℓ_to, field_to) in keys(op_to.index_map)
            if field_to != :T
                continue  # Shear goes into toroidal equation
            end
            if ℓ_to < m_pert_to
                continue
            end

            # Compute spherical harmonic coupling
            coupling_coeff = compute_sh_coupling_coefficient(
                ℓ_from, m_from, ℓ_bs, m_bs, ℓ_to, m_to
            )

            if abs(coupling_coeff) < 1e-14
                continue
            end

            idx_from = op_from.index_map[(ℓ_from, :P)]
            idx_to = op_to.index_map[(ℓ_to, :T)]

            from_offset = first(idx_from) - 1
            to_offset = first(idx_to) - 1

            # Coupling: L_from/r² × ∂ū_φ/∂r
            shear_coeff = L_from .* duphi_dr_bs ./ (r.^2)

            for i in 1:Nr
                row = to_offset + i
                col = from_offset + i
                if row <= size(C, 1) && col <= size(C, 2)
                    C[row, col] += coupling_coeff * shear_coeff[i]
                end
            end
        end
    end
end


"""
    compute_sh_coupling_coefficient(ℓ1, m1, ℓ2, m2, ℓ3, m3)

Compute the spherical harmonic coupling coefficient for the product:
    Y_{ℓ1,m1} × Y_{ℓ2,m2} → Y_{ℓ3,m3}

This uses the Gaunt coefficient (integral of three spherical harmonics).
Selection rules require:
- m1 + m2 = m3 (azimuthal selection)
- |ℓ1 - ℓ2| ≤ ℓ3 ≤ ℓ1 + ℓ2 (triangle inequality)
- ℓ1 + ℓ2 + ℓ3 even (parity)
"""
function compute_sh_coupling_coefficient(ℓ1::Int, m1::Int, ℓ2::Int, m2::Int, ℓ3::Int, m3::Int)
    # Selection rule: m1 + m2 = m3
    if m1 + m2 != m3
        return 0.0
    end

    # Triangle inequality
    if !(abs(ℓ1 - ℓ2) <= ℓ3 <= ℓ1 + ℓ2)
        return 0.0
    end

    # Parity selection
    if (ℓ1 + ℓ2 + ℓ3) % 2 != 0
        return 0.0
    end

    # Check m constraints
    if abs(m1) > ℓ1 || abs(m2) > ℓ2 || abs(m3) > ℓ3
        return 0.0
    end

    # Compute Gaunt coefficient using simplified formula
    # For small ℓ values, use analytical approximation
    norm_factor = sqrt((2*ℓ1 + 1) * (2*ℓ2 + 1) * (2*ℓ3 + 1) / (4*π))

    # Wigner 3j symbol (0 0 0)
    w3j_000 = wigner3j_simple(ℓ1, ℓ2, ℓ3, 0, 0, 0)

    # Wigner 3j symbol (m1 m2 -m3)
    w3j_mmm = wigner3j_simple(ℓ1, ℓ2, ℓ3, m1, m2, -m3)

    gaunt = norm_factor * w3j_000 * w3j_mmm

    return gaunt
end


"""
    wigner3j_simple(j1, j2, j3, m1, m2, m3)

Compute Wigner 3j symbol using WignerSymbols.jl package.

The Wigner 3j symbol is:
    (j1  j2  j3)
    (m1  m2  m3)

This is a wrapper around WignerSymbols.wigner3j for validated computation.
"""
function wigner3j_simple(j1::Int, j2::Int, j3::Int, m1::Int, m2::Int, m3::Int)
    # Use the validated WignerSymbols.jl package
    return Float64(WignerSymbols.wigner3j(j1, j2, j3, m1, m2, m3))
end


"""
    assemble_block_matrices(problem, single_mode_ops, coupling_ops, verbose)

Assemble the full block-coupled matrices A_coupled and B_coupled.

The structure is:
    ┌                     ┐
    │ A_{m1}   C_{12}  0  │
    │ C_{21}   A_{m2} C_{23}│
    │ 0      C_{32}  A_{m3}│
    └                     ┘

where A_{mi} are single-mode operators and C_{ij} are coupling operators.
"""
function assemble_block_matrices(problem::CoupledModeProblem{T},
                                  single_mode_ops::Dict,
                                  coupling_ops::Dict,
                                  verbose::Bool) where T
    n_total = problem.total_dofs
    A_coupled = zeros(ComplexF64, n_total, n_total)
    B_coupled = zeros(ComplexF64, n_total, n_total)

    # Fill in diagonal blocks (single-mode operators)
    for m in problem.m_range
        A_m, B_m, _ = single_mode_ops[m]
        block_range = problem.block_indices[m]

        A_coupled[block_range, block_range] .= A_m
        B_coupled[block_range, block_range] .= B_m
    end

    # Fill in off-diagonal blocks (coupling operators)
    # Currently empty due to diagonal approximation
    for ((m_from, m_to), C) in coupling_ops
        if !isempty(C)
            range_from = problem.block_indices[m_from]
            range_to = problem.block_indices[m_to]
            A_coupled[range_to, range_from] .+= C
        end
    end

    if verbose
        nnz_A = count(!iszero, A_coupled)
        nnz_B = count(!iszero, B_coupled)
        sparsity_A = 100.0 * (1.0 - nnz_A / length(A_coupled))
        sparsity_B = 100.0 * (1.0 - nnz_B / length(B_coupled))

        println("  Matrix size: $(n_total) × $(n_total)")
        println("  A sparsity:  $(round(sparsity_A, digits=1))%")
        println("  B sparsity:  $(round(sparsity_B, digits=1))%")
    end

    return A_coupled, B_coupled
end


"""
    solve_block_eigenvalue_problem(A, B, σ_target, nev, verbose)

Solve the generalized eigenvalue problem A x = λ B x using shift-invert.

Uses KrylovKit for iterative solution.
"""
function solve_block_eigenvalue_problem(A::Matrix{ComplexF64},
                                         B::Matrix{ComplexF64},
                                         σ_target::Real,
                                         nev::Int,
                                         verbose::Bool)
    n = size(A, 1)

    # Shift-invert: (A - σ B)^{-1} B
    # Eigenvalues of this operator are 1/(λ - σ), so λ = σ + 1/μ
    # Use a small imaginary shift to avoid singularity from boundary conditions
    shift = ComplexF64(σ_target) + 1e-6im
    A_shifted = A - shift * B

    if verbose
        println("  Factorizing shifted matrix (size $(n))...")
    end

    # Factor the shifted matrix with pivoting
    F = lu(A_shifted, check=false)

    # Check if factorization succeeded
    if !issuccess(F)
        if verbose
            println("  Warning: LU factorization may be near-singular, trying with regularization...")
        end
        # Add small regularization to diagonal
        for i in 1:n
            A_shifted[i, i] += 1e-12
        end
        F = lu(A_shifted)
    end

    if verbose
        println("  Running Krylov iteration...")
    end

    # Define the linear map for shift-invert
    function shift_invert_map(x)
        return F \ (B * x)
    end

    # Create a random initial vector
    x0 = randn(ComplexF64, n)

    # Use KrylovKit to find eigenvalues
    # We want the largest magnitude eigenvalues of the shift-invert operator
    vals, vecs, info = KrylovKit.eigsolve(
        shift_invert_map,
        x0, nev, :LM;
        krylovdim = max(2*nev + 10, 30),
        tol = 1e-8,
        maxiter = 200
    )

    if verbose
        println("  Converged: $(info.converged) eigenvalues")
    end

    # Transform back: λ = σ + 1/μ
    eigenvalues = [shift + 1.0/μ for μ in vals]

    # Sort by real part (descending)
    perm = sortperm(real.(eigenvalues), rev=true)
    eigenvalues = eigenvalues[perm]
    eigenvectors = hcat([vecs[i] for i in perm]...)

    return eigenvalues, eigenvectors
end


# =============================================================================
#  Main Solver Functions
# =============================================================================

"""
    solve_triglobal_eigenvalue_problem(params::TriGlobalParams{T};
                                       σ_target=0.0, nev=6, verbose=true) where T

Solve the tri-global eigenvalue problem to find growth rates and eigenmodes.

This solves the block-coupled eigenvalue problem:
    A_coupled x = λ B_coupled x

where different azimuthal modes m couple through the non-axisymmetric basic state.

Arguments:
- `params` - Tri-global parameters
- `σ_target` - Target growth rate for shift-invert (default: 0.0)
- `nev` - Number of eigenvalues to compute (default: 6)
- `verbose` - Print progress information (default: true)

Returns:
- `eigenvalues` - Complex growth rates λ = σ + iω (sorted by real part, descending)
- `eigenvectors` - Corresponding eigenmodes (columns of matrix)
"""
function solve_triglobal_eigenvalue_problem(params::TriGlobalParams{T};
                                            σ_target=0.0, nev=6, verbose=true) where T
    # Setup problem structure
    problem = setup_coupled_mode_problem(params)

    if verbose
        println("="^70)
        println("Tri-Global Eigenvalue Problem")
        println("="^70)
        println("  Mode range:        ", problem.m_range)
        println("  Basic state modes: ", problem.all_m_bs)
        println("  Total DOFs:        ", problem.total_dofs)
        println("  Target eigenvalues:", nev)
        println()
    end

    # Step 1: Build single-mode operators for each m
    if verbose
        println("Building single-mode operators for each m...")
    end

    single_mode_ops = build_single_mode_operators(problem, verbose)

    # Step 2: Build coupling operators between modes
    if verbose
        println("\nBuilding mode coupling operators...")
    end

    coupling_ops = build_mode_coupling_operators(problem, single_mode_ops, verbose)

    # Step 3: Assemble block-coupled matrices
    if verbose
        println("\nAssembling block-coupled matrices...")
    end

    A_coupled, B_coupled = assemble_block_matrices(problem, single_mode_ops, coupling_ops, verbose)

    # Step 4: Solve eigenvalue problem
    if verbose
        println("\nSolving eigenvalue problem (shift-invert, σ=$σ_target)...")
    end

    eigenvalues, eigenvectors = solve_block_eigenvalue_problem(
        A_coupled, B_coupled, σ_target, nev, verbose
    )

    if verbose
        println("\n" * "="^70)
        println("Eigenvalue Results:")
        println("="^70)
        for (i, λ) in enumerate(eigenvalues[1:min(nev, length(eigenvalues))])
            σ = real(λ)
            ω = imag(λ)
            println(@sprintf("  %2d: σ = %+.6e, ω = %+.6e", i, σ, ω))
        end
        println()
    end

    return eigenvalues, eigenvectors
end


"""
    find_critical_rayleigh_triglobal(E, Pr, χ, m_range, lmax, Nr,
                                     basic_state_3d;
                                     Ra_min=1e5, Ra_max=1e8,
                                     tol=1e-4, max_iter=20,
                                     mechanical_bc=:no_slip,
                                     thermal_bc=:fixed_temperature,
                                     verbose=true)

Find critical Rayleigh number for onset on a 3D basic state (tri-global analysis).

Uses bisection to find Ra_c where the leading growth rate σ = 0.

Arguments:
- `E` - Ekman number
- `Pr` - Prandtl number
- `χ` - Radius ratio
- `m_range` - Range of perturbation modes (e.g., -2:2)
- `lmax` - Maximum spherical harmonic degree
- `Nr` - Number of radial points
- `basic_state_3d` - The 3D basic state (BasicState3D)
- `Ra_min` - Lower bound for Ra search (default: 1e5)
- `Ra_max` - Upper bound for Ra search (default: 1e8)
- `tol` - Tolerance for bisection (default: 1e-4)
- `max_iter` - Maximum iterations (default: 20)
- `mechanical_bc` - Boundary conditions (default: :no_slip)
- `thermal_bc` - Thermal boundary conditions (default: :fixed_temperature)
- `verbose` - Print progress (default: true)

Returns:
- `Ra_c` - Critical Rayleigh number
- `σ_c` - Growth rate at Ra_c (should be ≈ 0)
- `ω_c` - Drift frequency at Ra_c
"""
function find_critical_rayleigh_triglobal(E, Pr, χ, m_range, lmax, Nr,
                                          basic_state_3d;
                                          Ra_min=1e5, Ra_max=1e8,
                                          tol=1e-4, max_iter=20,
                                          mechanical_bc=:no_slip,
                                          thermal_bc=:fixed_temperature,
                                          verbose=true)
    if verbose
        println("="^70)
        println("Finding Critical Rayleigh Number (Tri-Global)")
        println("="^70)
        println("  E           = ", @sprintf("%.2e", E))
        println("  Pr          = ", Pr)
        println("  χ           = ", χ)
        println("  m_range     = ", m_range)
        println("  lmax        = ", lmax)
        println("  Nr          = ", Nr)
        println("  Tolerance   = ", tol)
        println("  Max iter    = ", max_iter)
        println()
    end

    # Bisection algorithm
    Ra_low = Ra_min
    Ra_high = Ra_max

    # Test bounds
    if verbose
        println("Testing bounds...")
    end

    params_low = TriGlobalParams(
        E=E, Pr=Pr, Ra=Ra_low, χ=χ, m_range=m_range, lmax=lmax, Nr=Nr,
        basic_state_3d=basic_state_3d,
        mechanical_bc=mechanical_bc, thermal_bc=thermal_bc
    )
    vals_low, _ = solve_triglobal_eigenvalue_problem(params_low; nev=3, verbose=false)
    σ_low = real(vals_low[1])

    params_high = TriGlobalParams(
        E=E, Pr=Pr, Ra=Ra_high, χ=χ, m_range=m_range, lmax=lmax, Nr=Nr,
        basic_state_3d=basic_state_3d,
        mechanical_bc=mechanical_bc, thermal_bc=thermal_bc
    )
    vals_high, _ = solve_triglobal_eigenvalue_problem(params_high; nev=3, verbose=false)
    σ_high = real(vals_high[1])

    if verbose
        println("  Ra = $(Ra_low):  σ = $(σ_low)")
        println("  Ra = $(Ra_high): σ = $(σ_high)")
        println()
    end

    if σ_low > 0
        @warn "Lower bound Ra=$Ra_low is already unstable (σ=$σ_low > 0)"
        if verbose
            println("  Returning lower bound as estimate.")
        end
        return Ra_low, σ_low, imag(vals_low[1])
    end

    if σ_high < 0
        @warn "Upper bound Ra=$Ra_high is still stable (σ=$σ_high < 0)"
        if verbose
            println("  Returning upper bound as estimate.")
        end
        return Ra_high, σ_high, imag(vals_high[1])
    end

    # Bisection loop
    if verbose
        println("Starting bisection...")
        println(@sprintf("  %-4s  %-12s  %-12s  %-12s", "Iter", "Ra", "σ", "Δ Ra"))
        println("  " * "-"^45)
    end

    for iter in 1:max_iter
        Ra_mid = 0.5 * (Ra_low + Ra_high)

        params_mid = TriGlobalParams(
            E=E, Pr=Pr, Ra=Ra_mid, χ=χ, m_range=m_range, lmax=lmax, Nr=Nr,
            basic_state_3d=basic_state_3d,
            mechanical_bc=mechanical_bc, thermal_bc=thermal_bc
        )
        vals_mid, _ = solve_triglobal_eigenvalue_problem(params_mid; nev=3, verbose=false)
        σ_mid = real(vals_mid[1])
        ω_mid = imag(vals_mid[1])

        Delta_Ra = Ra_high - Ra_low

        if verbose
            println(@sprintf("  %-4d  %-12.6e  %+-.6e  %-12.6e",
                           iter, Ra_mid, σ_mid, Delta_Ra))
        end

        # Check convergence
        if abs(σ_mid) < tol * abs(σ_low) || Delta_Ra < tol * Ra_mid
            if verbose
                println()
                println("  Converged!")
                println("  Ra_c = ", @sprintf("%.6e", Ra_mid))
                println("  σ_c  = ", @sprintf("%+.6e", σ_mid))
                println("  ω_c  = ", @sprintf("%+.6e", ω_mid))
            end
            return Ra_mid, σ_mid, ω_mid
        end

        # Update bounds
        if σ_mid > 0
            Ra_high = Ra_mid
        else
            Ra_low = Ra_mid
        end
    end

    # Max iterations reached
    Ra_mid = 0.5 * (Ra_low + Ra_high)
    params_mid = TriGlobalParams(
        E=E, Pr=Pr, Ra=Ra_mid, χ=χ, m_range=m_range, lmax=lmax, Nr=Nr,
        basic_state_3d=basic_state_3d,
        mechanical_bc=mechanical_bc, thermal_bc=thermal_bc
    )
    vals_mid, _ = solve_triglobal_eigenvalue_problem(params_mid; nev=3, verbose=false)
    σ_mid = real(vals_mid[1])
    ω_mid = imag(vals_mid[1])

    @warn "Maximum iterations ($max_iter) reached without full convergence"
    if verbose
        println("  Returning best estimate:")
        println("  Ra_c = ", @sprintf("%.6e", Ra_mid))
        println("  σ_c  = ", @sprintf("%+.6e", σ_mid))
    end

    return Ra_mid, σ_mid, ω_mid
end
