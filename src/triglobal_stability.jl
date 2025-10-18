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
#  Placeholder functions for future implementation
# =============================================================================

"""
    solve_triglobal_eigenvalue_problem(params::TriGlobalParams{T};
                                       σ_target=0.0, nev=6) where T

Solve the tri-global eigenvalue problem to find growth rates and eigenmodes.

This is a PLACEHOLDER for the full implementation, which requires:
1. Assembling the block-coupled operator matrices
2. Including advection terms (ū·∇)θ', (u'·∇)θ̄, etc.
3. Solving the large sparse eigenvalue problem
4. Extracting growth rates σ and drift frequencies ω

Arguments:
- `params` - Tri-global parameters
- `σ_target` - Target growth rate (for shift-invert)
- `nev` - Number of eigenvalues to compute

Returns:
- `eigenvalues` - Complex growth rates λ = σ + iω
- `eigenvectors` - Corresponding eigenmodes

Status: TODO - Requires significant implementation effort
"""
function solve_triglobal_eigenvalue_problem(params::TriGlobalParams{T};
                                            σ_target=0.0, nev=6) where T
    # Setup problem structure
    problem = setup_coupled_mode_problem(params)

    println("Tri-global problem setup:")
    println("  Mode range: ", problem.m_range)
    println("  Total DOFs: ", problem.total_dofs)
    println("  Basic state modes m_bs: ", problem.all_m_bs)
    println()
    println("Mode coupling structure:")
    for m in problem.m_range
        println("  m=$m couples to: ", problem.coupling_graph[m])
    end
    println()

    # TODO: Implement the actual eigenvalue solve
    # This requires:
    # 1. Building single-mode operators for each m (using existing OnsetParams logic)
    # 2. Building coupling operators between modes m and m±m_bs
    # 3. Assembling block-sparse matrices
    # 4. Using iterative eigensolvers (KrylovKit, Arpack)

    @warn "solve_triglobal_eigenvalue_problem is not yet fully implemented!"
    @warn "This is a framework for future development."

    return nothing, nothing
end


"""
    find_critical_rayleigh_triglobal(E, Pr, χ, m_range, lmax, Nr,
                                     basic_state_3d; kwargs...)

Find critical Rayleigh number for onset on a 3D basic state (tri-global analysis).

This extends find_critical_rayleigh to handle non-axisymmetric basic states
where multiple perturbation modes couple.

Status: TODO - Placeholder for future implementation
"""
function find_critical_rayleigh_triglobal(E, Pr, χ, m_range, lmax, Nr,
                                          basic_state_3d; kwargs...)
    @warn "find_critical_rayleigh_triglobal is not yet implemented!"
    @warn "This requires completing solve_triglobal_eigenvalue_problem first."

    return nothing, nothing, nothing
end
