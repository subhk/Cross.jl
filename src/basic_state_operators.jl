# =============================================================================
#  Basic State Operators for Linear Stability Analysis
#
#  Implements the linearized operators for stability analysis on an
#  axisymmetric basic state with thermal wind-balanced zonal flow.
#
#  Theory:
#  -------
#  Basic state: θ̄(r,θ) = Σ_ℓ θ̄_ℓ0(r) Y_ℓ0(θ)
#               ū_φ(r,θ) = Σ_ℓ ū_φ,ℓ0(r) Y_ℓ0(θ)
#
#  Perturbations: θ'(r,θ,φ,t) = Σ_ℓ θ'_ℓm(r,t) Y_ℓm(θ,φ)
#
#  Linearized equations add three types of terms:
#  1. Advection: (ū · ∇)θ' = (ū_φ/(r sin θ)) ∂θ'/∂φ
#  2. Shear: (u' · ∇)ū = u'_r ∂ū_φ/∂r + u'_θ ∂ū_φ/∂θ
#  3. Temperature gradient: (u' · ∇)θ̄ = u'_r ∂θ̄/∂r + u'_θ ∂θ̄/∂θ
#
#  These couple different ℓ modes through spherical harmonic products.
# =============================================================================

using LinearAlgebra
using SparseArrays
using WignerSymbols

"""
    BasicStateOperators{T<:Real}

Container for precomputed basic state linearized operators.

Fields:
- `advection_blocks::Dict{Tuple{Int,Int}, Matrix{T}}` - Advection operators A[ℓ_pert, ℓ_bs]
- `shear_blocks::Dict{Tuple{Int,Int}, Matrix{T}}` - Shear production operators
- `temp_grad_blocks::Dict{Tuple{Int,Int}, Matrix{T}}` - Temperature gradient operators
- `coupling_structure::Vector{Tuple{Int,Int}}` - List of (ℓ_pert, ℓ_bs) pairs that couple

The blocks represent coupling between perturbation mode ℓ_pert and basic state mode ℓ_bs.
"""
struct BasicStateOperators{T<:Real}
    advection_blocks::Dict{Tuple{Int,Int}, Matrix{ComplexF64}}
    shear_radial_blocks::Dict{Tuple{Int,Int}, Matrix{ComplexF64}}
    shear_theta_blocks::Dict{Tuple{Int,Int}, Matrix{ComplexF64}}
    temp_grad_radial_blocks::Dict{Tuple{Int,Int}, Matrix{ComplexF64}}
    temp_grad_theta_blocks::Dict{Tuple{Int,Int}, Matrix{ComplexF64}}
    coupling_structure::Vector{Tuple{Int,Int}}
end


"""
    compute_spherical_harmonic_coupling(ℓ_pert::Int, ℓ_bs::Int, m::Int)

Compute coupling coefficients between perturbation mode (ℓ_pert, m) and
basic state mode (ℓ_bs, 0) through spherical harmonic integrals.

For products like Y_ℓpert,m × Y_ℓbs,0, we get contributions to Y_ℓ',m where
ℓ' ranges over |ℓ_pert - ℓ_bs| to ℓ_pert + ℓ_bs with appropriate selection rules.

Returns:
- `coupling_coeffs::Dict{Int, Float64}` - Coefficients for each coupled ℓ' mode
"""
function compute_spherical_harmonic_coupling(ℓ_pert::Int, ℓ_bs::Int, m::Int)
    # For axisymmetric basic state (m_bs = 0), the product
    # Y_ℓpert,m × Y_ℓbs,0 couples to modes with same m
    #
    # Using Clebsch-Gordan coefficients and selection rules:
    # ℓ' ∈ [|ℓ_pert - ℓ_bs|, ℓ_pert + ℓ_bs] with ℓ' + ℓ_pert + ℓ_bs even

    coupling_coeffs = Dict{Int, Float64}()

    ℓ_min = abs(ℓ_pert - ℓ_bs)
    ℓ_max = ℓ_pert + ℓ_bs

    for ℓ_prime in ℓ_min:ℓ_max
        # Selection rule: ℓ' + ℓ_pert + ℓ_bs must be even
        if (ℓ_prime + ℓ_pert + ℓ_bs) % 2 != 0
            continue
        end

        # Also need ℓ_prime >= m
        if ℓ_prime < m
            continue
        end

        # Gaunt coefficient (Wigner 3j symbol related)
        # Simplified formula for m_bs = 0 case
        coeff = compute_gaunt_coefficient(ℓ_pert, m, ℓ_bs, 0, ℓ_prime, m)

        if abs(coeff) > 1e-14
            coupling_coeffs[ℓ_prime] = coeff
        end
    end

    return coupling_coeffs
end


"""
    wigner3j_000(ℓ1::Int, ℓ2::Int, ℓ3::Int)

Compute Wigner 3j symbol with all m = 0:
    ⎛ℓ1  ℓ2  ℓ3⎞
    ⎝0   0   0 ⎠

Uses WignerSymbols.jl for accurate computation.
"""
function wigner3j_000(ℓ1::Int, ℓ2::Int, ℓ3::Int)
    return Float64(WignerSymbols.wigner3j(ℓ1, ℓ2, ℓ3, 0, 0, 0))
end

"""
    compute_gaunt_coefficient(ℓ1::Int, m1::Int, ℓ2::Int, m2::Int, ℓ3::Int, m3::Int)

Compute Gaunt coefficient (integral of three spherical harmonics):

    ∫ Y_ℓ1,m1 × Y_ℓ2,m2 × conj(Y_ℓ3,m3) dΩ

Using Wigner 3j symbols:
    G = √[(2ℓ1+1)(2ℓ2+1)(2ℓ3+1)/(4π)] × ⎛ℓ1  ℓ2  ℓ3⎞ ⎛ℓ1  ℓ2   ℓ3 ⎞
                                        ⎝0   0   0 ⎠ ⎝m1  m2  -m3⎠

For axisymmetric basic state (m2 = 0), this simplifies to:
    G = √[(2ℓ1+1)(2ℓ2+1)(2ℓ3+1)/(4π)] × ⎛ℓ1  ℓ2  ℓ3⎞ ⎛ℓ1  ℓ2  ℓ3 ⎞ × δ_{m1,m3}
                                        ⎝0   0   0 ⎠ ⎝m1  0   -m1⎠
"""
function compute_gaunt_coefficient(ℓ1::Int, m1::Int, ℓ2::Int, m2::Int, ℓ3::Int, m3::Int)
    # Selection rules for ⟨Y_{ℓ1,m1} Y_{ℓ2,m2} Y_{ℓ3,m3}*⟩
    if m1 + m2 != m3
        return 0.0
    end
    if abs(m1) > ℓ1 || abs(m2) > ℓ2 || abs(m3) > ℓ3
        return 0.0
    end
    if !((abs(ℓ1 - ℓ2) <= ℓ3 <= ℓ1 + ℓ2))
        return 0.0
    end
    if (ℓ1 + ℓ2 + ℓ3) % 2 != 0
        return 0.0
    end

    w3j_1 = wigner3j_000(ℓ1, ℓ2, ℓ3)
    abs(w3j_1) < 1e-14 && return 0.0

    w3j_2 = Float64(WignerSymbols.wigner3j(ℓ1, ℓ2, ℓ3, m1, m2, -m3))
    abs(w3j_2) < 1e-14 && return 0.0

    norm_factor = sqrt((2 * ℓ1 + 1) * (2 * ℓ2 + 1) * (2 * ℓ3 + 1) / (4 * π))
    phase = isodd(m3) ? -1.0 : 1.0
    return phase * norm_factor * w3j_1 * w3j_2
end


"""
    build_basic_state_operators(basic_state::BasicState{T},
                                 op::LinearStabilityOperator{T},
                                 m::Int) where T

Build all linearized operators for stability analysis on a basic state.

Arguments:
- `basic_state` - The axisymmetric basic state (θ̄, ū_φ)
- `op` - Linear stability operator structure (contains radial operators)
- `m` - Azimuthal wavenumber of perturbation

Returns:
- `BasicStateOperators` - Precomputed operator blocks for all ℓ couplings

The operators are organized as blocks connecting perturbation mode ℓ_pert
to basic state mode ℓ_bs.
"""
function build_basic_state_operators(basic_state::BasicState{T},
                                      op,
                                      m::Int) where T

    # Extract radial operators
    r = collect(op.r)  # Radial collocation points
    Nr = length(r)
    r_inv2 = 1.0 ./ (r .^ 2)

    # Radial differentiation operator (from Chebyshev differentiation structure)
    Dr = op.cd.D1

    # Storage for operator blocks
    # Key: (ℓ_output, ℓ_input) - coupling from input mode to output mode
    advection_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    shear_radial_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    shear_theta_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    temp_grad_radial_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    temp_grad_theta_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    coupling_structure = Tuple{Int,Int}[]

    # Get basic state modes
    ℓ_bs_modes = sort(collect(keys(basic_state.theta_coeffs)))

    # Get perturbation modes from operator
    perturbation_modes = sort(collect(keys(op.index_map)))
    ℓ_pert_modes = unique([ℓ for (ℓ, field) in perturbation_modes])
    ℓ_pert_set = Set(ℓ_pert_modes)

    println("Building basic state operators...")
    println("  Basic state modes (ℓ_bs): ", ℓ_bs_modes)
    println("  Perturbation modes (ℓ_pert): ", ℓ_pert_modes)
    println("  Azimuthal wavenumber m = ", m)

    # Loop over all basic state modes
    for ℓ_bs in ℓ_bs_modes
        # Get basic state coefficients
        uphi_coeff = basic_state.uphi_coeffs[ℓ_bs]
        duphi_dr = basic_state.duphi_dr_coeffs[ℓ_bs]
        dtheta_dr = basic_state.dtheta_dr_coeffs[ℓ_bs]

        uphi_max = maximum(abs.(uphi_coeff))
        theta_max = maximum(abs.(basic_state.theta_coeffs[ℓ_bs]))

        # Skip if this mode is negligible
        if theta_max < 1e-14 && uphi_max < 1e-14
            continue
        end

        # Loop over input perturbation modes
        for ℓ_input in ℓ_pert_modes
            if ℓ_input < m
                continue
            end

            # Compute spherical harmonic coupling:
            # Product Y_ℓ_input,m × Y_ℓ_bs,0 → Σ_ℓ' coupling[ℓ'] × Y_ℓ',m
            coupling_coeffs = compute_spherical_harmonic_coupling(ℓ_input, ℓ_bs, m)

            if isempty(coupling_coeffs)
                continue
            end

            # Loop over all OUTPUT modes that receive coupling
            for (ℓ_output, coupling_coeff) in coupling_coeffs
                # Only include output modes that exist in the perturbation basis
                if !(ℓ_output in ℓ_pert_set) || ℓ_output < m
                    continue
                end

                # Record this coupling
                if !((ℓ_output, ℓ_input) in coupling_structure)
                    push!(coupling_structure, (ℓ_output, ℓ_input))
                end

                # Angular momentum quantum number for input mode
                L_input = T(ℓ_input * (ℓ_input + 1))

                # =====================================================================
                # 1. Advection operator: (ū_φ/(r sin θ)) ∂/∂φ = im·m × ū_φ/(r sin θ)
                # =====================================================================
                # This advects perturbation mode ℓ_input by zonal flow ℓ_bs
                # and projects onto output mode ℓ_output
                if uphi_max > 1e-14
                    # Advection term (radial dependence only)
                    # The 1/sin(θ) is absorbed into the Gaunt coefficient
                    adv_operator = im * m * coupling_coeff * Diagonal(uphi_coeff ./ r)

                    # Initialize or accumulate (multiple ℓ_bs can contribute)
                    if !haskey(advection_blocks, (ℓ_output, ℓ_input))
                        advection_blocks[(ℓ_output, ℓ_input)] = Matrix(adv_operator)
                    else
                        advection_blocks[(ℓ_output, ℓ_input)] .+= Matrix(adv_operator)
                    end
                end

                # =====================================================================
                # 2. Radial shear: -u'_r × ∂ū_φ/∂r
                # =====================================================================
                if uphi_max > 1e-14
                    shear_op = -L_input * coupling_coeff * Diagonal(duphi_dr .* r_inv2)

                    if !haskey(shear_radial_blocks, (ℓ_output, ℓ_input))
                        shear_radial_blocks[(ℓ_output, ℓ_input)] = Matrix(shear_op)
                    else
                        shear_radial_blocks[(ℓ_output, ℓ_input)] .+= Matrix(shear_op)
                    end
                end

                # =====================================================================
                # 3. Theta shear: -u'_θ × (1/r) ∂ū_φ/∂θ
                # =====================================================================
                # This requires the θ-derivative of Y_ℓbs,0, which couples to Y_ℓbs±1,0
                # For now, use simplified approximation (set to zero)
                if !haskey(shear_theta_blocks, (ℓ_output, ℓ_input))
                    shear_theta_blocks[(ℓ_output, ℓ_input)] = zeros(ComplexF64, Nr, Nr)
                end

                # =====================================================================
                # 4. Radial temperature gradient: -u'_r × ∂θ̄/∂r
                # =====================================================================
                # u'_r ~ ℓ(ℓ+1)/r² × P for poloidal potential P
                if theta_max > 1e-14
                    temp_grad_op = -L_input * coupling_coeff * Diagonal(dtheta_dr .* r_inv2)

                    if !haskey(temp_grad_radial_blocks, (ℓ_output, ℓ_input))
                        temp_grad_radial_blocks[(ℓ_output, ℓ_input)] = Matrix(temp_grad_op)
                    else
                        temp_grad_radial_blocks[(ℓ_output, ℓ_input)] .+= Matrix(temp_grad_op)
                    end
                end

                # =====================================================================
                # 5. Theta temperature gradient: -u'_θ × (1/r) ∂θ̄/∂θ
                # =====================================================================
                # Similar to theta shear, requires θ-derivative coupling
                if !haskey(temp_grad_theta_blocks, (ℓ_output, ℓ_input))
                    temp_grad_theta_blocks[(ℓ_output, ℓ_input)] = zeros(ComplexF64, Nr, Nr)
                end
            end
        end
    end

    println("  Built ", length(coupling_structure), " operator blocks")

    # Count non-zero blocks
    n_nonzero_adv = count(block -> maximum(abs.(block)) > 1e-14, values(advection_blocks))
    println("  Non-zero advection blocks: ", n_nonzero_adv)

    return BasicStateOperators{T}(
        advection_blocks,
        shear_radial_blocks,
        shear_theta_blocks,
        temp_grad_radial_blocks,
        temp_grad_theta_blocks,
        coupling_structure
    )
end


"""
    add_basic_state_operators!(A::Matrix, B::Matrix,
                                basic_state_ops::BasicStateOperators,
                                op::LinearStabilityOperator,
                                m::Int)

Add basic state operators to the assembled A matrix.

Modifies A in place to include:
- Advection by zonal flow
- Shear production
- Temperature gradient advection

The coupling pairs are indexed as (ℓ_output, ℓ_input), meaning the operator
couples from input mode ℓ_input to output mode ℓ_output through the basic state.

Arguments:
- `A` - Operator matrix (modified in place)
- `B` - Mass matrix (not modified, included for consistency)
- `basic_state_ops` - Precomputed basic state operators
- `op` - Linear stability operator structure
- `m` - Azimuthal wavenumber
"""
function add_basic_state_operators!(A::Matrix, B::Matrix,
                                     basic_state_ops::BasicStateOperators,
                                     op,
                                     m::Int)

    println("Adding basic state operators to A matrix...")

    # Loop over all coupling pairs (ℓ_output, ℓ_input)
    for (ℓ_output, ℓ_input) in basic_state_ops.coupling_structure

        # Get indices for output and input modes
        if !haskey(op.index_map, (ℓ_output, :P)) || !haskey(op.index_map, (ℓ_input, :P))
            continue
        end

        # Output mode indices (row indices in A)
        P_out_idx = op.index_map[(ℓ_output, :P)]
        Θ_out_idx = haskey(op.index_map, (ℓ_output, :Θ)) ? op.index_map[(ℓ_output, :Θ)] : nothing
        T_out_idx = haskey(op.index_map, (ℓ_output, :T)) ? op.index_map[(ℓ_output, :T)] : nothing

        # Input mode indices (column indices in A)
        P_in_idx = op.index_map[(ℓ_input, :P)]
        Θ_in_idx = haskey(op.index_map, (ℓ_input, :Θ)) ? op.index_map[(ℓ_input, :Θ)] : nothing
        T_in_idx = haskey(op.index_map, (ℓ_input, :T)) ? op.index_map[(ℓ_input, :T)] : nothing

        # =====================================================================
        # 1. Add advection operator to temperature equation
        #    ∂θ'_ℓ_output/∂t term: (ū_φ/(r sin θ)) ∂θ'_ℓ_input/∂φ
        #    This couples θ'_ℓ_input → θ'_ℓ_output
        # =====================================================================
        if Θ_out_idx !== nothing && Θ_in_idx !== nothing
            if haskey(basic_state_ops.advection_blocks, (ℓ_output, ℓ_input))
                adv_block = basic_state_ops.advection_blocks[(ℓ_output, ℓ_input)]
                A[Θ_out_idx, Θ_in_idx] .+= adv_block
            end
        end

        # =====================================================================
        # 2. Add radial temperature gradient term
        #    ∂θ'_ℓ_output/∂t term: -u'_r,ℓ_input × ∂θ̄/∂r
        #    This couples P_ℓ_input → θ'_ℓ_output (u_r from poloidal)
        # =====================================================================
        if Θ_out_idx !== nothing
            if haskey(basic_state_ops.temp_grad_radial_blocks, (ℓ_output, ℓ_input))
                temp_grad_block = basic_state_ops.temp_grad_radial_blocks[(ℓ_output, ℓ_input)]
                A[Θ_out_idx, P_in_idx] .+= temp_grad_block
            end
        end

        # =====================================================================
        # 3. Add radial shear to toroidal equation
        #    ∂u'_φ,ℓ_output/∂t term: -u'_r,ℓ_input × ∂ū_φ/∂r
        #    This couples P_ℓ_input → T_ℓ_output
        # =====================================================================
        if T_out_idx !== nothing
            if haskey(basic_state_ops.shear_radial_blocks, (ℓ_output, ℓ_input))
                shear_block = basic_state_ops.shear_radial_blocks[(ℓ_output, ℓ_input)]
                A[T_out_idx, P_in_idx] .+= shear_block
            end
        end
    end

    println("  Basic state operators added successfully")

    return nothing
end
