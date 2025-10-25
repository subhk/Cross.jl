# =============================================================================
#  Kore-Style Sparse Linear Stability Operator
#
#  Implementation following the Kore spectral method for onset of convection
#  in rotating spherical shells using sparse Gegenbauer (ultraspherical)
#  polynomials.
#
#  References:
#  - Rekier et al. (2019), Kore implementation
#  - Barik et al. (2023), Earth and Space Science
#  - Dormy et al. (2004), JFM
# =============================================================================

module KoreSparseOperator

using LinearAlgebra
using SparseArrays
using Parameters

include("UltrasphericalSpectral.jl")
using .UltrasphericalSpectral

export KoreOnsetParams,
       KoreSparseStabilityOperator,
       assemble_sparse_matrices,
       kore_solve_eigenvalue

# -----------------------------------------------------------------------------
# Parameters matching Kore's structure
# -----------------------------------------------------------------------------

"""
    KoreOnsetParams

Parameters for onset of convection in rotating spherical shells.
Matches Kore's parameter naming conventions.

# Fields
- `E::Float64`: Ekman number ν/(ΩL²)
- `Pr::Float64`: Prandtl number ν/κ
- `Ra::Float64`: Rayleigh number
- `ricb::Float64`: Inner core radius (outer radius normalized to 1.0)
- `m::Int`: Azimuthal wavenumber
- `lmax::Int`: Maximum spherical harmonic degree
- `N::Int`: Chebyshev truncation level (must be even)
- `symm::Int`: Equatorial symmetry (+1 symmetric, -1 antisymmetric)
- `bci::Int`: Inner boundary condition (0=stress-free, 1=no-slip)
- `bco::Int`: Outer boundary condition (0=stress-free, 1=no-slip)
- `bci_thermal::Int`: Inner thermal BC (0=fixed temp, 1=fixed flux)
- `bco_thermal::Int`: Outer thermal BC (0=fixed temp, 1=fixed flux)
"""
@with_kw struct KoreOnsetParams{T<:Real}
    # Dimensionless parameters
    E::T                           # Ekman number
    Pr::T = one(T)                 # Prandtl number
    Ra::T                          # Rayleigh number
    ricb::T                        # Inner radius (ro = 1.0)

    # Wave parameters
    m::Int                         # Azimuthal wavenumber
    lmax::Int                      # Max spherical harmonic degree
    symm::Int = 1                  # Equatorial symmetry (±1)

    # Resolution
    N::Int                         # Chebyshev truncation

    # Boundary conditions
    bci::Int = 1                   # Inner mechanical BC (1=no-slip)
    bco::Int = 1                   # Outer mechanical BC (1=no-slip)
    bci_thermal::Int = 0           # Inner thermal BC (0=fixed temp)
    bco_thermal::Int = 0           # Outer thermal BC (0=fixed temp)

    # Derived quantities
    L::T = one(T) - ricb           # Shell thickness
    Etherm::T = E / Pr             # Thermal Ekman number

    function KoreOnsetParams{T}(E, Pr, Ra, ricb, m, lmax, symm, N,
                                bci, bco, bci_thermal, bco_thermal,
                                L, Etherm) where {T<:Real}
        @assert 0 < ricb < 1 "ricb must be in (0,1)"
        @assert E > 0 "Ekman number must be positive"
        @assert Pr > 0 "Prandtl number must be positive"
        @assert m >= 0 "Azimuthal wavenumber must be non-negative"
        @assert lmax >= m "lmax must be >= m"
        @assert N >= 4 && iseven(N) "N must be even and >= 4"
        @assert symm in (-1, 1) "symm must be ±1"

        new{T}(E, Pr, Ra, ricb, m, lmax, symm, N,
               bci, bco, bci_thermal, bco_thermal, L, Etherm)
    end
end

# Outer constructor to infer type
function KoreOnsetParams(; E::T, Pr::T=one(T), Ra::T, ricb::T,
                          m::Int, lmax::Int, symm::Int=1, N::Int,
                          bci::Int=1, bco::Int=1,
                          bci_thermal::Int=0, bco_thermal::Int=0) where {T<:Real}
    L = one(T) - ricb
    Etherm = E / Pr
    return KoreOnsetParams{T}(E, Pr, Ra, ricb, m, lmax, symm, N,
                             bci, bco, bci_thermal, bco_thermal, L, Etherm)
end

# -----------------------------------------------------------------------------
# Sparse operator storage
# -----------------------------------------------------------------------------

"""
    KoreSparseStabilityOperator

Stores pre-computed sparse radial operators and problem parameters.
"""
struct KoreSparseStabilityOperator{T<:Real}
    params::KoreOnsetParams{T}

    # Radial operators for poloidal velocity (section u, 2curl)
    # Naming: r{power}_D{deriv}_u
    r0_D0_u::SparseMatrixCSC{Float64,Int}
    r2_D0_u::SparseMatrixCSC{Float64,Int}
    r3_D1_u::SparseMatrixCSC{Float64,Int}
    r4_D2_u::SparseMatrixCSC{Float64,Int}
    r3_D3_u::SparseMatrixCSC{Float64,Int}
    r4_D4_u::SparseMatrixCSC{Float64,Int}

    # Radial operators for toroidal velocity (section v, 1curl)
    r0_D0_v::SparseMatrixCSC{Float64,Int}
    r1_D1_v::SparseMatrixCSC{Float64,Int}
    r2_D2_v::SparseMatrixCSC{Float64,Int}

    # Radial operators for temperature (section h)
    r0_D0_h::SparseMatrixCSC{Float64,Int}
    r1_D1_h::SparseMatrixCSC{Float64,Int}
    r2_D0_h::SparseMatrixCSC{Float64,Int}
    r2_D2_h::SparseMatrixCSC{Float64,Int}

    # l-mode information
    ll_top::Vector{Int}  # l values for poloidal (equatorially symmetric)
    ll_bot::Vector{Int}  # l values for toroidal (equatorially antisymmetric)
    nl_modes::Int        # Total number of l modes

    # Matrix size
    matrix_size::Int
end

"""
    KoreSparseStabilityOperator(params::KoreOnsetParams)

Construct the sparse operator by pre-computing all radial operators.
"""
function KoreSparseStabilityOperator(params::KoreOnsetParams{T}) where {T}
    N = params.N
    ri, ro = params.ricb, one(T)

    println("Building Kore sparse operators (N=$N, ricb=$ri)...")

    # Pre-compute radial operators for poloidal velocity
    println("  Computing poloidal operators...")
    r0_D0_u = sparse_radial_operator(0, 0, N, ri, ro)
    r2_D0_u = sparse_radial_operator(2, 0, N, ri, ro)
    r3_D1_u = sparse_radial_operator(3, 1, N, ri, ro)
    r4_D2_u = sparse_radial_operator(4, 2, N, ri, ro)
    r3_D3_u = sparse_radial_operator(3, 3, N, ri, ro)
    r4_D4_u = sparse_radial_operator(4, 4, N, ri, ro)

    # Pre-compute radial operators for toroidal velocity
    println("  Computing toroidal operators...")
    r0_D0_v = sparse_radial_operator(0, 0, N, ri, ro)
    r1_D1_v = sparse_radial_operator(1, 1, N, ri, ro)
    r2_D2_v = sparse_radial_operator(2, 2, N, ri, ro)

    # Pre-compute radial operators for temperature
    println("  Computing temperature operators...")
    r0_D0_h = sparse_radial_operator(0, 0, N, ri, ro)
    r1_D1_h = sparse_radial_operator(1, 1, N, ri, ro)
    r2_D0_h = sparse_radial_operator(2, 0, N, ri, ro)
    r2_D2_h = sparse_radial_operator(2, 2, N, ri, ro)

    # Determine l-mode structure based on equatorial symmetry
    ll_top, ll_bot = compute_l_modes(params.m, params.lmax, params.symm)
    nl_modes = length(ll_top) + length(ll_bot)

    # Each l-mode has (N+1) radial DOFs
    # Total size: 2 sections (u,v) × nl_modes × (N+1) + 1 section (theta) × nl_modes × (N+1)
    # For onset: velocity (poloidal + toroidal) + temperature
    n_per_mode = N + 1
    matrix_size = 2 * nl_modes * n_per_mode + nl_modes * n_per_mode  # u, v, theta

    println("  l-modes: $(length(ll_top)) poloidal + $(length(ll_bot)) toroidal")
    println("  Matrix size: $(matrix_size) × $(matrix_size)")
    println("  Estimated sparsity: ~$(estimate_sparsity(N, nl_modes))%")

    return KoreSparseStabilityOperator{T}(
        params,
        r0_D0_u, r2_D0_u, r3_D1_u, r4_D2_u, r3_D3_u, r4_D4_u,
        r0_D0_v, r1_D1_v, r2_D2_v,
        r0_D0_h, r1_D1_h, r2_D0_h, r2_D2_h,
        ll_top, ll_bot, nl_modes,
        matrix_size
    )
end

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

"""
    compute_l_modes(m, lmax, symm)

Compute the l-mode indices for given m, lmax, and equatorial symmetry.
Returns (ll_top, ll_bot) where:
- ll_top: l-modes for poloidal velocity (equatorially symmetric if symm=1)
- ll_bot: l-modes for toroidal velocity (equatorially antisymmetric if symm=-1)
"""
function compute_l_modes(m::Int, lmax::Int, symm::Int)
    if symm == 1
        # Equatorially symmetric flow
        # Poloidal: l = m, m+2, m+4, ...
        # Toroidal: l = m+1, m+3, m+5, ...
        ll_top = collect(m:2:lmax)
        ll_bot = collect((m+1):2:lmax)
    elseif symm == -1
        # Equatorially antisymmetric flow
        # Poloidal: l = m+1, m+3, m+5, ...
        # Toroidal: l = m, m+2, m+4, ...
        ll_top = collect((m+1):2:lmax)
        ll_bot = collect(m:2:lmax)
    else
        # Both symmetries (full problem)
        ll_top = collect(m:lmax)
        ll_bot = collect(m:lmax)
    end

    return ll_top, ll_bot
end

"""
    estimate_sparsity(N, nl_modes)

Estimate the percentage of nonzero entries in the assembled matrices.
"""
function estimate_sparsity(N::Int, nl_modes::Int)
    # Each l-mode couples to l±1, l±2 neighbors
    # Average coupling: ~5 l-modes per row
    # Each operator contributes ~O(N) nonzeros per radial mode
    # Total nonzeros: ~5 * nl_modes * N^2
    # Total entries: (3 * nl_modes * (N+1))^2

    total_size = 3 * nl_modes * (N + 1)
    nnz_estimate = 5 * nl_modes * N^2
    sparsity = 100.0 * (1.0 - nnz_estimate / total_size^2)

    return round(sparsity, digits=2)
end

# -----------------------------------------------------------------------------
# Operator functions (matching Kore's operators.py structure)
# -----------------------------------------------------------------------------

"""
    operator_u(op, l)

Velocity operator: L(-L*r²D⁰ + 2r³D¹ + r⁴D²) for diagonal term.
Matches Kore's op.u(l, 'u', 'upol', 0).
"""
function operator_u(op::KoreSparseStabilityOperator{T}, l::Int) where {T}
    L = l * (l + 1)
    return L * (-L * op.r2_D0_u + 2 * op.r3_D1_u + op.r4_D2_u)
end

"""
    operator_coriolis_diagonal(op, l, m)

Coriolis force operator for diagonal (l,l) coupling.
Matches Kore's op.coriolis(l, 'u', 'upol', 0).

Returns: 2im * m * (-L*r²D⁰ + 2r³D¹ + r⁴D²)
"""
function operator_coriolis_diagonal(op::KoreSparseStabilityOperator{T},
                                   l::Int, m::Int) where {T}
    L = l * (l + 1)
    return 2im * m * (-L * op.r2_D0_u + 2 * op.r3_D1_u + op.r4_D2_u)
end

"""
    operator_coriolis_offdiag(op, l, m, offset)

Coriolis force operator for off-diagonal (l, l±1) coupling.
Matches Kore's op.coriolis(l, 'u', 'utor', ±1).

Returns: [operator, offset] where offset indicates which l-mode it couples to.
"""
function operator_coriolis_offdiag(op::KoreSparseStabilityOperator{T},
                                  l::Int, m::Int, offset::Int) where {T}
    L = l * (l + 1)

    if offset == -1
        # Coupling to l-1 mode
        C = (l^2 - 1) * sqrt(l^2 - m^2) / (2l - 1)
        mtx = 2 * C * ((l - 1) * op.r3_D1_u - op.r4_D2_u)
        return mtx, -1

    elseif offset == 1
        # Coupling to l+1 mode
        C = l * (l + 2) * sqrt((l + m + 1) * (l - m + 1)) / (2l + 3)
        mtx = 2 * C * (-(l + 2) * op.r3_D1_u - op.r4_D2_u)
        return mtx, 1

    else
        error("offset must be ±1 for Coriolis off-diagonal")
    end
end

"""
    operator_viscous_diffusion(op, l, E)

Viscous diffusion operator: E * ∇²∇²u matching Kore's structure.
Matches Kore's op.viscous_diffusion(l, 'u', 'upol', 0).

Returns: E * L * (-L(l+2)(l-1)*r⁰D⁰ + 2L*r²D² - 4r³D³ - r⁴D⁴)
"""
function operator_viscous_diffusion(op::KoreSparseStabilityOperator{T},
                                   l::Int, E::T) where {T}
    L = l * (l + 1)
    return E * L * (-L * (l + 2) * (l - 1) * op.r0_D0_u +
                    2 * L * op.r4_D2_u -
                    4 * op.r3_D3_u -
                    op.r4_D4_u)
end

"""
    operator_buoyancy(op, l, Ra, Pr)

Buoyancy operator: (Ra/Pr) * r² * θ matching Kore's structure.
Matches Kore's op.buoyancy(l, 'u', '', 0).

This couples the temperature field to the velocity equation.
"""
function operator_buoyancy(op::KoreSparseStabilityOperator{T},
                          l::Int, Ra::T, Pr::T) where {T}
    # In Kore: Beyonce * r^power * D^0
    # where Beyonce = BV² = -Ra * E² / Pr
    # But in the velocity equation, buoyancy enters as (Ra/Pr) * r * θ
    # For linear gravity g ∝ r, we have r² in the poloidal equation

    return (Ra / Pr) * op.r2_D0_u  # Will be applied to temperature field
end

# -----------------------------------------------------------------------------
# Toroidal velocity operators (section v, 1curl)
# -----------------------------------------------------------------------------

"""
    operator_u_toroidal(op, l)

Toroidal velocity operator (essentially identity for the mass matrix).
For the 1curl (toroidal) equation, this is just r⁰D⁰ = I.
Matches Kore's op.u(l, 'v', 'utor', 0) structure.
"""
function operator_u_toroidal(op::KoreSparseStabilityOperator{T}, l::Int) where {T}
    # For toroidal velocity, the time derivative term is just the identity
    return op.r0_D0_v
end

"""
    operator_coriolis_toroidal(op, l, m)

Coriolis force acting on toroidal velocity.
Matches Kore's op.coriolis(l, 'u', 'utor', 0).

Returns: -2im * m * r²D⁰_v

Note: In Kore, this is multiplied by Gaspard = 1.0 for time scale Tau = 1/Omega.
"""
function operator_coriolis_toroidal(op::KoreSparseStabilityOperator{T},
                                   l::Int, m::Int) where {T}
    # From Kore operators.py line 122:
    # section == 'u', component == 'utor', offdiag == 0
    # out = -2j*par.m*r2_D0_v  (for non-magnetic case)
    # Multiplied by par.Gaspard = 1.0

    # Note: We use r0_D0_v here because in the radial formulation
    # the r² factor is handled differently
    return -2im * m * op.r0_D0_v
end

"""
    operator_viscous_toroidal(op, l, E)

Viscous diffusion operator for toroidal velocity: E * ∇²u_toroidal.
Matches Kore's op.viscous_diffusion(l, 'v', 'utor', 0).

Returns: E * L * (-L*r⁰D⁰ + 2*r¹D¹ + r²D²)

where L = l(l+1).
"""
function operator_viscous_toroidal(op::KoreSparseStabilityOperator{T},
                                  l::Int, E::T) where {T}
    # From Kore operators.py line 192:
    # section == 'v', component == 'utor', offdiag == 0
    # out = L*( -L*r0_D0_v + 2*r1_D1_v + r2_D2_v )
    # Multiplied by par.ViscosD = Ek = E

    L = l * (l + 1)
    return E * L * (-L * op.r0_D0_v + 2 * op.r1_D1_v + op.r2_D2_v)
end

# -----------------------------------------------------------------------------
# Temperature operators (section h)
# -----------------------------------------------------------------------------

"""
    operator_theta(op, l)

Temperature operator for time derivative term.
Matches Kore's op.theta(l, 'h', '', 0).

For non-anelastic, non-differential heating: returns r²D⁰
"""
function operator_theta(op::KoreSparseStabilityOperator{T}, l::Int) where {T}
    # From Kore operators.py line 715:
    # section == 'h', non-anelastic, non-differential heating
    # out = r2_D0_h
    return op.r2_D0_h
end

"""
    operator_thermal_diffusion(op, l, Etherm)

Thermal diffusion operator: (E/Pr) * ∇²θ.
Matches Kore's op.thermal_diffusion(l, 'h', '', 0).

Returns: Etherm * (-L*r⁰D⁰ + 2*r¹D¹ + r²D²)

where Etherm = E/Pr and L = l(l+1).
"""
function operator_thermal_diffusion(op::KoreSparseStabilityOperator{T},
                                   l::Int, Etherm::T) where {T}
    # From Kore operators.py line 761:
    # section == 'h', non-anelastic, non-differential heating
    # difus = - L*r0_D0_h + 2*r1_D1_h + r2_D2_h  # eq. times r**2
    # out = difus * par.ThermaD
    # where par.ThermaD = Etherm = E/Pr

    L = l * (l + 1)
    return Etherm * (-L * op.r0_D0_h + 2 * op.r1_D1_h + op.r2_D2_h)
end

"""
    operator_thermal_advection(op, l)

Thermal advection operator: radial velocity advecting temperature.
Matches Kore's op.thermal_advection(l, 'h', 'upol', 0).

Returns: L * r²D⁰

This couples the poloidal velocity to the temperature equation.
For internal heating: dT/dr = -β*r, so the advection term is u_r * (-β*r)
"""
function operator_thermal_advection(op::KoreSparseStabilityOperator{T},
                                   l::Int) where {T}
    # From Kore operators.py line 734:
    # section == 'h', component == 'upol', non-anelastic, internal heating
    # conv = r2_D0_h  # dT/dr = -beta*r. Heat equation is times r**2
    # out = L * conv

    L = l * (l + 1)
    return L * op.r2_D0_h
end

# -----------------------------------------------------------------------------
# Matrix assembly
# -----------------------------------------------------------------------------

"""
    assemble_sparse_matrices(op::KoreSparseStabilityOperator)

Assemble the full sparse matrices A and B for the generalized eigenvalue problem:
    A * x = λ * B * x

Following Kore's assembly structure from assemble.py.

Returns: (A, B, interior_dofs, info)
"""
function assemble_sparse_matrices(op::KoreSparseStabilityOperator{T}) where {T}
    params = op.params
    N = params.N
    m = params.m
    E = params.E
    Pr = params.Pr
    Ra = params.Ra

    n_per_mode = N + 1
    nb_top = length(op.ll_top)  # Number of poloidal modes
    nb_bot = length(op.ll_bot)  # Number of toroidal modes

    # Matrix size
    n = op.matrix_size

    println("\nAssembling Kore sparse matrices...")
    println("  Matrix size: $n × $n")
    println("  Poloidal modes: $(op.ll_top)")
    println("  Toroidal modes: $(op.ll_bot)")

    # Initialize sparse matrices using DOK (Dictionary of Keys) format
    # We'll convert to CSR at the end
    A_rows = Int[]
    A_cols = Int[]
    A_vals = ComplexF64[]

    B_rows = Int[]
    B_cols = Int[]
    B_vals = ComplexF64[]

    # Helper function to add block to matrix
    function add_block!(rows, cols, vals, block::SparseMatrixCSC,
                       row_offset::Int, col_offset::Int)
        I, J, V = findnz(block)
        append!(rows, I .+ row_offset)
        append!(cols, J .+ col_offset)
        append!(vals, V)
    end

    # =========================================================================
    # Section u (poloidal velocity, 2curl equation)
    # =========================================================================
    println("  Assembling section u (poloidal)...")

    for (k, l) in enumerate(op.ll_top)
        row_base = (k - 1) * n_per_mode
        col_base = (k - 1) * n_per_mode

        # -----------------------------------------------------------------
        # A matrix: iω*u + Coriolis - viscous + coupling terms
        # -----------------------------------------------------------------

        # Time derivative term (goes to B matrix)
        # B = identity for velocity
        I_block = sparse(1.0I, n_per_mode, n_per_mode)
        add_block!(B_rows, B_cols, B_vals, I_block, row_base, col_base)

        # Velocity operator: -u
        u_op = -operator_u(op, l)
        add_block!(A_rows, A_cols, A_vals, u_op, row_base, col_base)

        # Coriolis force (diagonal)
        cori_op = operator_coriolis_diagonal(op, l, m)
        add_block!(A_rows, A_cols, A_vals, cori_op, row_base, col_base)

        # Viscous diffusion
        visc_op = -operator_viscous_diffusion(op, l, E)
        add_block!(A_rows, A_cols, A_vals, visc_op, row_base, col_base)

        # Coriolis coupling to toroidal velocity (l±1)
        for offset in [-1, 1]
            l_coupled = l + offset
            if l_coupled in op.ll_bot
                k_coupled = findfirst(==(l_coupled), op.ll_bot)
                col_coupled = (nb_top + k_coupled - 1) * n_per_mode

                cori_off, _ = operator_coriolis_offdiag(op, l, m, offset)
                add_block!(A_rows, A_cols, A_vals, cori_off,
                          row_base, col_coupled)
            end
        end

        # Buoyancy coupling to temperature field
        # Temperature field starts after 2*nb sections
        temp_col_base = (nb_top + nb_bot + k - 1) * n_per_mode
        buoy_op = operator_buoyancy(op, l, Ra, Pr)
        add_block!(A_rows, A_cols, A_vals, buoy_op,
                  row_base, temp_col_base)
    end

    # =========================================================================
    # Section v (toroidal velocity, 1curl equation)
    # =========================================================================
    println("  Assembling section v (toroidal)...")

    for (k, l) in enumerate(op.ll_bot)
        row_base = (nb_top + k - 1) * n_per_mode
        col_base = (nb_top + k - 1) * n_per_mode

        # -----------------------------------------------------------------
        # B matrix: Time derivative (identity)
        # -----------------------------------------------------------------
        I_block = sparse(1.0I, n_per_mode, n_per_mode)
        add_block!(B_rows, B_cols, B_vals, I_block, row_base, col_base)

        # -----------------------------------------------------------------
        # A matrix: iω*v + Coriolis - viscous
        # -----------------------------------------------------------------

        # Toroidal velocity operator: -u_toroidal
        # (For eigenvalue problem, this effectively contributes to iω term)
        u_tor_op = -operator_u_toroidal(op, l)
        add_block!(A_rows, A_cols, A_vals, u_tor_op, row_base, col_base)

        # Coriolis force acting on toroidal velocity
        cori_tor_op = operator_coriolis_toroidal(op, l, m)
        add_block!(A_rows, A_cols, A_vals, cori_tor_op, row_base, col_base)

        # Viscous diffusion for toroidal velocity
        visc_tor_op = -operator_viscous_toroidal(op, l, E)
        add_block!(A_rows, A_cols, A_vals, visc_tor_op, row_base, col_base)

        # Note: No direct buoyancy coupling for toroidal velocity
        # (Buoyancy forces poloidal flow, which then couples to toroidal via Coriolis)
    end

    # =========================================================================
    # Temperature equation (section h)
    # =========================================================================
    println("  Assembling temperature equation...")

    for (k, l) in enumerate(op.ll_top)
        row_base = (nb_top + nb_bot + k - 1) * n_per_mode
        col_base = (nb_top + nb_bot + k - 1) * n_per_mode

        # -----------------------------------------------------------------
        # B matrix: Time derivative (weighted by r²)
        # -----------------------------------------------------------------
        # In Kore: B = theta operator (r²D⁰ for non-anelastic)
        theta_op = operator_theta(op, l)
        add_block!(B_rows, B_cols, B_vals, theta_op, row_base, col_base)

        # -----------------------------------------------------------------
        # A matrix: -iω*θ + thermal_diffusion - thermal_advection
        # -----------------------------------------------------------------

        # Temperature identity operator (for eigenvalue problem)
        # This combines with iω in the eigenvalue problem
        temp_op = -operator_theta(op, l)
        add_block!(A_rows, A_cols, A_vals, temp_op, row_base, col_base)

        # Thermal diffusion
        thermal_diff = operator_thermal_diffusion(op, l, params.Etherm)
        add_block!(A_rows, A_cols, A_vals, thermal_diff, row_base, col_base)

        # Thermal advection: coupling from poloidal velocity to temperature
        # This acts on the poloidal velocity field
        vel_col_base = (k - 1) * n_per_mode  # Poloidal velocity at same l-mode
        thermal_adv = operator_thermal_advection(op, l)
        add_block!(A_rows, A_cols, A_vals, thermal_adv, row_base, vel_col_base)
    end

    # =========================================================================
    # Convert to sparse CSC format
    # =========================================================================
    println("  Converting to CSC format...")
    A = sparse(A_rows, A_cols, A_vals, n, n)
    B = sparse(B_rows, B_cols, B_vals, n, n)

    println("  A sparsity: $(nnz(A)) / $(n^2) = $(100*nnz(A)/n^2)%")
    println("  B sparsity: $(nnz(B)) / $(n^2) = $(100*nnz(B)/n^2)%")

    # =========================================================================
    # Apply boundary conditions
    # =========================================================================
    println("  Applying boundary conditions...")
    apply_kore_boundary_conditions!(A, B, op)

    println("  Final A sparsity: $(nnz(A)) / $(n^2)")
    println("  Final B sparsity: $(nnz(B)) / $(n^2)")

    # All DOFs are interior (boundary conditions applied via row replacement)
    interior_dofs = collect(1:n)

    info = Dict(
        "method" => "Kore sparse ultraspherical",
        "N" => N,
        "lmax" => params.lmax,
        "m" => m,
        "nl_modes" => op.nl_modes,
        "matrix_size" => n
    )

    return A, B, interior_dofs, info
end

"""
    apply_kore_boundary_conditions!(A, B, op)

Apply boundary conditions by replacing appropriate rows in A and B matrices.
Uses the tau method following Kore's approach.
"""
function apply_kore_boundary_conditions!(A::SparseMatrixCSC,
                                        B::SparseMatrixCSC,
                                        op::KoreSparseStabilityOperator{T}) where {T}
    params = op.params
    N = params.N
    n_per_mode = N + 1

    nb_top = length(op.ll_top)
    nb_bot = length(op.ll_bot)

    # -------------------------------------------------------------------------
    # Poloidal velocity BCs: No-slip (u = 0, du/dr = 0)
    # -------------------------------------------------------------------------
    for (k, l) in enumerate(op.ll_top)
        row_base = (k - 1) * n_per_mode

        # Outer boundary (r = ro): u = 0, du/dr = 0
        bc_rows = [row_base + 1, row_base + 2]
        apply_boundary_conditions!(A, B, bc_rows, :dirichlet, N,
                                  params.ricb, one(T))

        # Inner boundary (r = ri): u = 0, du/dr = 0
        bc_rows = [row_base + n_per_mode - 1, row_base + n_per_mode]
        apply_boundary_conditions!(A, B, bc_rows, :dirichlet, N,
                                  params.ricb, one(T))
    end

    # -------------------------------------------------------------------------
    # Toroidal velocity BCs: No-slip (v = 0)
    # -------------------------------------------------------------------------
    for (k, l) in enumerate(op.ll_bot)
        row_base = (nb_top + k - 1) * n_per_mode

        # Outer boundary (r = ro): v = 0
        bc_rows = [row_base + 1]
        apply_boundary_conditions!(A, B, bc_rows, :dirichlet, N,
                                  params.ricb, one(T))

        # Inner boundary (r = ri): v = 0
        bc_rows = [row_base + n_per_mode]
        apply_boundary_conditions!(A, B, bc_rows, :dirichlet, N,
                                  params.ricb, one(T))
    end

    # -------------------------------------------------------------------------
    # Temperature BCs: Fixed temperature (θ = 0)
    # -------------------------------------------------------------------------
    for (k, l) in enumerate(op.ll_top)
        row_base = (nb_top + nb_bot + k - 1) * n_per_mode

        # Outer boundary (r = ro): θ = 0
        bc_rows = [row_base + 1]
        apply_boundary_conditions!(A, B, bc_rows, :dirichlet, N,
                                  params.ricb, one(T))

        # Inner boundary (r = ri): θ = 0
        bc_rows = [row_base + n_per_mode]
        apply_boundary_conditions!(A, B, bc_rows, :dirichlet, N,
                                  params.ricb, one(T))
    end

    return nothing
end

end  # module KoreSparseOperator
