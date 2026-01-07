# =============================================================================
#  Sparse Linear Stability Operator
#
#  Implementation using ultraspherical spectral method for onset of convection
#  in rotating spherical shells using sparse Gegenbauer (ultraspherical)
#  polynomials.
#
#  References:
#  - Rekier et al. (2019), Reference implementation
#  - Barik et al. (2023), Earth and Space Science
#  - Dormy et al. (2004), JFM
# =============================================================================

module SparseOperator

using LinearAlgebra
using SparseArrays
using Parameters

include("UltrasphericalSpectral.jl")
using .UltrasphericalSpectral

export SparseOnsetParams,
       SparseStabilityOperator,
       assemble_sparse_matrices

# -----------------------------------------------------------------------------
# Parameters for sparse onset calculations
# -----------------------------------------------------------------------------

"""
    SparseOnsetParams

Parameters for onset of convection in rotating spherical shells.
Uses sparse ultraspherical spectral discretization.

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
@with_kw struct SparseOnsetParams{T<:Real}
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

    # Heating type
    heating::Symbol = :differential # :internal or :differential

    # Derived quantities
    L::T = one(T) - ricb           # Shell thickness
    Etherm::T = E / Pr             # Thermal Ekman number

    function SparseOnsetParams{T}(E, Pr, Ra, ricb, m, lmax, symm, N,
                                bci, bco, bci_thermal, bco_thermal, heating,
                                L, Etherm) where {T<:Real}
        @assert 0 < ricb < 1 "ricb must be in (0,1)"
        @assert E > 0 "Ekman number must be positive"
        @assert Pr > 0 "Prandtl number must be positive"
        @assert m >= 0 "Azimuthal wavenumber must be non-negative"
        @assert lmax >= m "lmax must be >= m"
        @assert N >= 4 && iseven(N) "N must be even and >= 4"
        @assert symm in (-1, 1) "symm must be ±1"
        @assert heating in (:internal, :differential) "heating must be :internal or :differential"

        new{T}(E, Pr, Ra, ricb, m, lmax, symm, N,
               bci, bco, bci_thermal, bco_thermal, heating, L, Etherm)
    end
end

# Outer constructor to infer type
function SparseOnsetParams(; E::T, Pr::T=one(T), Ra::T, ricb::T,
                          m::Int, lmax::Int, symm::Int=1, N::Int,
                          bci::Int=1, bco::Int=1,
                          bci_thermal::Int=0, bco_thermal::Int=0,
                          heating::Symbol=:differential) where {T<:Real}
    L = one(T) - ricb
    Etherm = E / Pr
    return SparseOnsetParams{T}(E, Pr, Ra, ricb, m, lmax, symm, N,
                             bci, bco, bci_thermal, bco_thermal, heating, L, Etherm)
end

# -----------------------------------------------------------------------------
# Sparse operator storage
# -----------------------------------------------------------------------------

"""
    SparseStabilityOperator

Stores pre-computed sparse radial operators and problem parameters.
"""
struct SparseStabilityOperator{T<:Real}
    params::SparseOnsetParams{T}

    # Radial operators for poloidal velocity (section u, 2curl)
    # Naming: r{power}_D{deriv}_u
    r0_D0_u::SparseMatrixCSC{Float64,Int}
    r2_D0_u::SparseMatrixCSC{Float64,Int}
    r2_D2_u::SparseMatrixCSC{Float64,Int}  # For viscous diffusion
    r3_D0_u::SparseMatrixCSC{Float64,Int}  # For Coriolis coupling
    r3_D1_u::SparseMatrixCSC{Float64,Int}
    r4_D0_u::SparseMatrixCSC{Float64,Int}  # For buoyancy coupling
    r4_D1_u::SparseMatrixCSC{Float64,Int}  # For Coriolis coupling
    r4_D2_u::SparseMatrixCSC{Float64,Int}
    r3_D3_u::SparseMatrixCSC{Float64,Int}
    r4_D4_u::SparseMatrixCSC{Float64,Int}

    # Radial operators for toroidal velocity (section v, 1curl)
    r0_D0_v::SparseMatrixCSC{Float64,Int}
    r1_D0_v::SparseMatrixCSC{Float64,Int}  # For toroidal-poloidal coupling
    r1_D1_v::SparseMatrixCSC{Float64,Int}
    r2_D0_v::SparseMatrixCSC{Float64,Int}  # For Coriolis (toroidal diagonal)
    r2_D1_v::SparseMatrixCSC{Float64,Int}  # For toroidal-poloidal coupling
    r2_D2_v::SparseMatrixCSC{Float64,Int}

    # Radial operators for temperature (section h)
    r0_D0_h::SparseMatrixCSC{Float64,Int}
    r1_D0_h::SparseMatrixCSC{Float64,Int}  # For differential heating
    r1_D1_h::SparseMatrixCSC{Float64,Int}
    r2_D0_h::SparseMatrixCSC{Float64,Int}
    r2_D1_h::SparseMatrixCSC{Float64,Int}  # For differential heating
    r2_D2_h::SparseMatrixCSC{Float64,Int}
    r3_D0_h::SparseMatrixCSC{Float64,Int}  # For differential heating
    r3_D2_h::SparseMatrixCSC{Float64,Int}  # For differential heating

    # l-mode information
    ll_top::Vector{Int}  # l values for poloidal (equatorially symmetric)
    ll_bot::Vector{Int}  # l values for toroidal (equatorially antisymmetric)
    nl_modes::Int        # Total number of l modes

    # Matrix size
    matrix_size::Int
end

"""
    SparseStabilityOperator(params::SparseOnsetParams)

Construct the sparse operator by pre-computing all radial operators.
"""
function SparseStabilityOperator(params::SparseOnsetParams{T}) where {T}
    N = params.N
    ri, ro = params.ricb, one(T)

    println("Building sparse operators (N=$N, ricb=$ri)...")

    # Pre-compute radial operators for poloidal velocity
    println("  Computing poloidal operators...")
    r0_D0_u = sparse_radial_operator(0, 0, N, ri, ro)
    r2_D0_u = sparse_radial_operator(2, 0, N, ri, ro)
    r2_D2_u = sparse_radial_operator(2, 2, N, ri, ro)  # For viscous diffusion
    r3_D0_u = sparse_radial_operator(3, 0, N, ri, ro)  # For Coriolis coupling
    r3_D1_u = sparse_radial_operator(3, 1, N, ri, ro)
    r4_D0_u = sparse_radial_operator(4, 0, N, ri, ro)  # For buoyancy coupling
    r4_D1_u = sparse_radial_operator(4, 1, N, ri, ro)  # For Coriolis coupling
    r4_D2_u = sparse_radial_operator(4, 2, N, ri, ro)
    r3_D3_u = sparse_radial_operator(3, 3, N, ri, ro)
    r4_D4_u = sparse_radial_operator(4, 4, N, ri, ro)

    # Pre-compute radial operators for toroidal velocity
    println("  Computing toroidal operators...")
    r0_D0_v = sparse_radial_operator(0, 0, N, ri, ro)
    r1_D0_v = sparse_radial_operator(1, 0, N, ri, ro)  # For toroidal-poloidal coupling
    r1_D1_v = sparse_radial_operator(1, 1, N, ri, ro)
    r2_D0_v = sparse_radial_operator(2, 0, N, ri, ro)  # For Coriolis (toroidal diagonal)
    r2_D1_v = sparse_radial_operator(2, 1, N, ri, ro)  # For toroidal-poloidal coupling
    r2_D2_v = sparse_radial_operator(2, 2, N, ri, ro)

    # Pre-compute radial operators for temperature
    println("  Computing temperature operators...")
    r0_D0_h = sparse_radial_operator(0, 0, N, ri, ro)
    r1_D0_h = sparse_radial_operator(1, 0, N, ri, ro)  # For differential heating
    r1_D1_h = sparse_radial_operator(1, 1, N, ri, ro)
    r2_D0_h = sparse_radial_operator(2, 0, N, ri, ro)
    r2_D1_h = sparse_radial_operator(2, 1, N, ri, ro)  # For differential heating
    r2_D2_h = sparse_radial_operator(2, 2, N, ri, ro)
    r3_D0_h = sparse_radial_operator(3, 0, N, ri, ro)  # For differential heating
    r3_D2_h = sparse_radial_operator(3, 2, N, ri, ro)  # For differential heating

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

    return SparseStabilityOperator{T}(
        params,
        r0_D0_u, r2_D0_u, r2_D2_u, r3_D0_u, r3_D1_u, r4_D0_u, r4_D1_u, r4_D2_u, r3_D3_u, r4_D4_u,
        r0_D0_v, r1_D0_v, r1_D1_v, r2_D0_v, r2_D1_v, r2_D2_v,
        r0_D0_h, r1_D0_h, r1_D1_h, r2_D0_h, r2_D1_h, r2_D2_h, r3_D0_h, r3_D2_h,
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
    @assert m >= 0 "m must be non-negative"
    @assert lmax >= m "lmax must be >= m"
    @assert symm in (-1, 0, 1) "symm must be -1, 0, or 1"

    # Following Kore's algorithm (bin/utils.py:174-183, function ell())
    # Key insight: For m=0, l-range is 1:(lmax+1) for spectral completeness
    #              For m>0, l-range is m:lmax as usual
    # This uses sign(m) to handle m=0 correctly

    if symm == 0
        # Both symmetries (full problem)
        if m == 0
            ll = collect(1:(lmax+1))
        else
            ll = collect(m:lmax)
        end
        return ll, ll
    end

    # Symmetric parameter: s=0 for antisymmetric, s=1 for symmetric
    s = div(symm + 1, 2)

    # Generate full l-range
    if m == 0
        ll = collect(1:(lmax+1))  # For m=0: l ∈ [1, lmax+1]
        sign_m = 0
    else
        ll = collect(m:lmax)      # For m>0: l ∈ [m, lmax]
        sign_m = 1
    end

    # Select modes based on parity (following Kore's indexing)
    # idp: indices for poloidal modes
    # idt: indices for toroidal modes
    idp_start = (sign_m + s) % 2 + 1      # Convert to 1-based indexing
    idt_start = (sign_m + s + 1) % 2 + 1

    ll_top = ll[idp_start:2:end]
    ll_bot = ll[idt_start:2:end]

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
# Operator functions (for rotating convection)
# -----------------------------------------------------------------------------

"""
    operator_u(op, l)

Poloidal velocity time derivative operator for the B matrix.
Returns L(L*r²D⁰ - 2r³D¹ - r⁴D²) where L = l(l+1).
Implements op.u(l, 'u', 'upol', 0) from Kore.

This represents r⁴·r̂·∇×∇×u (the 2curl operator weighted by r⁴).
Note: Signs are opposite to the Coriolis operator.
"""
function operator_u(op::SparseStabilityOperator{T}, l::Int) where {T}
    # From Kore line 35: out = L*( L*r2_D0_u - 2*r3_D1_u - r4_D2_u )
    # Note: Signs are OPPOSITE to Coriolis operator (line 63)
    # This represents r⁴·r̂·∇×∇×u (2curl of u, weighted by r⁴)
    L = l * (l + 1)
    return L * (L * op.r2_D0_u - 2 * op.r3_D1_u - op.r4_D2_u)
end

"""
    operator_coriolis_diagonal(op, l, m)

Coriolis force operator for diagonal (l,l) coupling.
Implements op.coriolis(l, 'u', 'upol', 0).

Returns: 2im * m * (-L*r²D⁰ + 2r³D¹ + r⁴D²)
"""
function operator_coriolis_diagonal(op::SparseStabilityOperator{T},
                                   l::Int, m::Int) where {T}
    L = l * (l + 1)
    return 2im * m * (-L * op.r2_D0_u + 2 * op.r3_D1_u + op.r4_D2_u)
end

"""
    operator_coriolis_offdiag(op, l, m, offset)

Coriolis force operator for off-diagonal (l, l±1) coupling.
Implements op.coriolis(l, 'u', 'utor', ±1).

Returns: [operator, offset] where offset indicates which l-mode it couples to.

Following Kore lines 68-86:
- For l-1: C = (l²-1)*sqrt(l²-m²)/(2l-1)
  out = 2*C*((l-1)*r³D⁰ - r⁴D¹)
- For l+1: C = l*(l+2)*sqrt((l+m+1)*(l-m+1))/(2l+3)
  out = 2*C*(-(l+2)*r³D⁰ - r⁴D¹)
"""
function operator_coriolis_offdiag(op::SparseStabilityOperator{T},
                                  l::Int, m::Int, offset::Int) where {T}
    if offset == -1
        # Coupling to l-1 mode
        C = (l^2 - 1) * sqrt(l^2 - m^2) / (2l - 1)
        mtx = 2 * C * ((l - 1) * op.r3_D0_u - op.r4_D1_u)
        return mtx, -1

    elseif offset == 1
        # Coupling to l+1 mode
        C = l * (l + 2) * sqrt((l + m + 1) * (l - m + 1)) / (2l + 3)
        mtx = 2 * C * (-(l + 2) * op.r3_D0_u - op.r4_D1_u)
        return mtx, 1

    else
        error("offset must be ±1 for Coriolis off-diagonal")
    end
end

"""
    operator_viscous_diffusion(op, l, E)

Viscous diffusion operator: E * ∇²∇²u for sparse spectral method.
Implements op.viscous_diffusion(l, 'u', 'upol', 0).

Returns: E * L * (-L(l+2)(l-1)*r⁰D⁰ + 2L*r²D² - 4r³D³ - r⁴D⁴)
"""
function operator_viscous_diffusion(op::SparseStabilityOperator{T},
                                   l::Int, E::T) where {T}
    L = l * (l + 1)
    return E * L * (-L * (l + 2) * (l - 1) * op.r0_D0_u +
                    2 * L * op.r2_D2_u -
                    4 * op.r3_D3_u -
                    op.r4_D4_u)
end

"""
    operator_buoyancy(op, l, Ra, Pr)

Buoyancy operator: beyonce * L * r⁴ * θ for sparse spectral method.
Implements op.buoyancy(l, 'u', '', 0).

This couples the temperature field to the poloidal velocity equation.

Following Kore and Cross.jl:
- beyonce = -Ra_internal * E² / Pr (the negative sign is crucial!)
- Ra_internal = Ra / gap³ for gap-based Rayleigh number conversion
- L = l(l+1)
- r⁴ weighting matches the poloidal equation (2curl, weighted by r⁴)

Note: The Ra provided is assumed to be gap-based (using shell thickness L = r_o - r_i
as length scale). We convert to internal Ra using Ra_internal = Ra / gap³ because
the non-dimensionalization uses r_o as the length scale.
Reference: Barik et al. (2023), Kore parameters.py
"""
function operator_buoyancy(op::SparseStabilityOperator{T},
                          l::Int, Ra::T, Pr::T) where {T}
    # Convert gap-based Ra to internal Ra
    # Ra_internal = Ra_gap / gap^3 (gap = r_o - r_i = 1 - ricb when r_o = 1)
    E = op.params.E
    ricb = op.params.ricb
    gap = one(T) - ricb
    Ra_internal = Ra / gap^3

    # Beyonce factor = BV² = -Ra_internal * E² / Pr
    beyonce = -Ra_internal * E^2 / Pr

    # L factor
    L = l * (l + 1)

    # Full buoyancy operator: beyonce * L * r⁴D⁰
    return beyonce * L * op.r4_D0_u
end

"""
    operator_coriolis_v_to_u(op, l, m, offset)

Coriolis force coupling from toroidal velocity (v) to poloidal velocity (u).
Implements op.coriolis(l, 'v', 'upol', ±1).

This is the REVERSE coupling to operator_coriolis_offdiag (which does u→v).
Both directions are required for correct rotating convection physics!

Following Kore operators.py lines 93-113:
- For l-1: C = (l²-1)*sqrt(l²-m²)/(2l-1)
  out = 2*C*((l-1)*r¹D⁰_v - r²D¹_v)
- For l+1: C = l*(l+2)*sqrt((l+m+1)*(l-m+1))/(2l+3)
  out = 2*C*(-(l+2)*r¹D⁰_v - r²D¹_v)

Returns: operator matrix
"""
function operator_coriolis_v_to_u(op::SparseStabilityOperator{T},
                                 l::Int, m::Int, offset::Int) where {T}
    if offset == -1
        # Coupling from v at mode l to u at mode l-1
        C = (l^2 - 1) * sqrt(l^2 - m^2) / (2l - 1)
        return 2 * C * ((l - 1) * op.r1_D0_v - op.r2_D1_v)

    elseif offset == 1
        # Coupling from v at mode l to u at mode l+1
        C = l * (l + 2) * sqrt((l + m + 1) * (l - m + 1)) / (2l + 3)
        return 2 * C * (-(l + 2) * op.r1_D0_v - op.r2_D1_v)

    else
        error("offset must be ±1 for Coriolis v→u coupling")
    end
end

# -----------------------------------------------------------------------------
# Toroidal velocity operators (section v, 1curl)
# -----------------------------------------------------------------------------

"""
    operator_u_toroidal(op, l)

Toroidal velocity time derivative operator for the B matrix.
Returns L * r²D⁰_v where L = l(l+1).
Implements op.u(l, 'v', 'utor', 0) from Kore.

The toroidal equation is weighted by r² (1curl equation), so all terms
including the time derivative must use r² weighting to maintain consistency.
"""
function operator_u_toroidal(op::SparseStabilityOperator{T}, l::Int) where {T}
    # For toroidal velocity time derivative operator
    # From Kore line 42: out = L*r2_D0_v  (section='v', component='utor', offdiag=0)
    # Comment: "r2* r.1curl(u)" - equation weighted by r²
    L = l * (l + 1)
    return L * op.r2_D0_v
end

"""
    operator_coriolis_toroidal(op, l, m)

Coriolis force acting on toroidal velocity.
Implements op.coriolis(l, 'u', 'utor', 0).

Returns: -2im * m * r²D⁰_v

Note: In Kore, this is multiplied by Gaspard = 1.0 for time scale Tau = 1/Omega.
"""
function operator_coriolis_toroidal(op::SparseStabilityOperator{T},
                                   l::Int, m::Int) where {T}
    # Following Kore line 121:
    # section == 'v', component == 'utor', offdiag == 0
    # out = -2j*par.m*r2_D0_v  (NO L factor!)
    # Multiplied by par.Gaspard = 1.0

    return -2im * m * op.r2_D0_v
end

"""
    operator_viscous_toroidal(op, l, E)

Viscous diffusion operator for toroidal velocity: E * ∇²u_toroidal.
Implements op.viscous_diffusion(l, 'v', 'utor', 0).

Returns: E * L * (-L*r⁰D⁰ + 2*r¹D¹ + r²D²)

where L = l(l+1).
"""
function operator_viscous_toroidal(op::SparseStabilityOperator{T},
                                  l::Int, E::T) where {T}
    # From reference implementation line 192:
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
Implements op.theta(l, 'h', '', 0).

Following Kore lines 712-716:
- For 'differential' heating: returns r³D⁰ (eq. times r³)
- For 'internal' heating: returns r²D⁰ (eq. times r²)
"""
function operator_theta(op::SparseStabilityOperator{T}, l::Int) where {T}
    if op.params.heating == :differential
        return op.r3_D0_h  # eq. times r³
    else  # :internal
        return op.r2_D0_h  # eq. times r²
    end
end

"""
    operator_thermal_diffusion(op, l, Etherm)

Thermal diffusion operator: (E/Pr) * ∇²θ.
Implements op.thermal_diffusion(l, 'h', '', 0).

Following Kore lines 758-761:
- For 'differential' heating: Etherm * (-L*r¹D⁰ + 2*r²D¹ + r³D²) (eq. times r³)
- For 'internal' heating: Etherm * (-L*r⁰D⁰ + 2*r¹D¹ + r²D²) (eq. times r²)

where Etherm = E/Pr and L = l(l+1).
"""
function operator_thermal_diffusion(op::SparseStabilityOperator{T},
                                   l::Int, Etherm::T) where {T}
    L = l * (l + 1)

    if op.params.heating == :differential
        # eq. times r³
        return Etherm * (-L * op.r1_D0_h + 2 * op.r2_D1_h + op.r3_D2_h)
    else  # :internal
        # eq. times r²
        return Etherm * (-L * op.r0_D0_h + 2 * op.r1_D1_h + op.r2_D2_h)
    end
end

"""
    operator_thermal_advection(op, l)

Thermal advection operator: radial velocity advecting temperature.
Implements op.thermal_advection(l, 'h', 'upol', 0).

Following Kore lines 733-740:
- For 'differential' heating: L * r⁰D⁰ * (ricb/gap), dT/dr = -β*r⁻², eq. times r³
- For 'internal' heating: L * r²D⁰, dT/dr = -β*r, eq. times r²

This couples the poloidal velocity to the temperature equation.
"""
function operator_thermal_advection(op::SparseStabilityOperator{T},
                                   l::Int) where {T}
    L = l * (l + 1)

    if op.params.heating == :differential
        # dT/dr = -beta * r⁻², eq. times r³
        ricb = op.params.ricb
        gap = one(T) - ricb
        return L * op.r0_D0_h * (ricb / gap)
    else  # :internal
        # dT/dr = -beta * r, eq. times r²
        return L * op.r2_D0_h
    end
end

# -----------------------------------------------------------------------------
# Matrix assembly
# -----------------------------------------------------------------------------

"""
    assemble_sparse_matrices(op::SparseStabilityOperator)

Assemble the full sparse matrices A and B for the generalized eigenvalue problem:
    A * x = λ * B * x

Assembly structure from assemble.py.

Returns: (A, B, interior_dofs, info)
"""
function assemble_sparse_matrices(op::SparseStabilityOperator{T}) where {T}
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

    println("\nAssembling sparse matrices...")
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
        # B matrix: Time derivative operator (negative per Cross.jl convention)
        # A matrix: RHS operators (Coriolis, viscous, buoyancy)
        # -----------------------------------------------------------------

        # Time derivative term: ∂u/∂t → -operator_u in B matrix (matches Kore)
        u_op = operator_u(op, l)
        add_block!(B_rows, B_cols, B_vals, -u_op, row_base, col_base)

        # Coriolis force (diagonal)
        cori_op = operator_coriolis_diagonal(op, l, m)
        add_block!(A_rows, A_cols, A_vals, cori_op, row_base, col_base)

        # Viscous diffusion (appears with a minus sign in Kore)
        visc_op = operator_viscous_diffusion(op, l, E)
        add_block!(A_rows, A_cols, A_vals, -visc_op, row_base, col_base)

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
        # B matrix: Time derivative operator (negative per Cross.jl convention)
        # A matrix: RHS operators (Coriolis, viscous)
        # -----------------------------------------------------------------

        # Time derivative term: ∂v/∂t → -operator_u_toroidal in B matrix (matches Kore)
        u_tor_op = operator_u_toroidal(op, l)
        add_block!(B_rows, B_cols, B_vals, -u_tor_op, row_base, col_base)

        # Coriolis force acting on toroidal velocity
        cori_tor_op = operator_coriolis_toroidal(op, l, m)
        add_block!(A_rows, A_cols, A_vals, cori_tor_op, row_base, col_base)

        # Viscous diffusion for toroidal velocity (minus sign per Kore)
        visc_tor_op = operator_viscous_toroidal(op, l, E)
        add_block!(A_rows, A_cols, A_vals, -visc_tor_op, row_base, col_base)

        # Coriolis coupling from toroidal to poloidal velocity (v → u, l±1)
        # This coupling goes in the TOROIDAL equation (v-rows), coupling to POLOIDAL variable (u-columns)
        # Physical meaning: Coriolis force in toroidal equation depends on poloidal velocity
        for offset in [-1, 1]
            l_coupled = l + offset
            if l_coupled in op.ll_top
                k_coupled = findfirst(==(l_coupled), op.ll_top)
                col_coupled = (k_coupled - 1) * n_per_mode  # Column for u at l_coupled

                # Compute Coriolis v→u coupling operator
                cori_v_to_u = operator_coriolis_v_to_u(op, l, m, offset)
                add_block!(A_rows, A_cols, A_vals, cori_v_to_u,
                          row_base, col_coupled)  # FIXED: v-rows (this equation), u-columns (coupled variable)

            end
        end

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
        # Reference: B = theta operator (r²D⁰ for non-anelastic)
        theta_op = operator_theta(op, l)
        add_block!(B_rows, B_cols, B_vals, theta_op, row_base, col_base)

        # -----------------------------------------------------------------
        # A matrix: RHS operators (thermal diffusion, advection)
        # -----------------------------------------------------------------

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
    apply_sparse_boundary_conditions!(A, B, op)

    println("  Final A sparsity: $(nnz(A)) / $(n^2)")
    println("  Final B sparsity: $(nnz(B)) / $(n^2)")

    # Identify interior DOFs (those with nonzero B diagonal after BCs)
    # Boundary conditions zero out rows in B, making it singular
    # For eigenvalue solving, we need only the interior DOFs
    B_diag = diag(B)
    interior_dofs = findall(i -> abs(B_diag[i]) > 1e-14, 1:n)
    println("  Interior DOFs: $(length(interior_dofs)) / $n")

    info = Dict(
        "method" => "Sparse ultraspherical",
        "N" => N,
        "lmax" => params.lmax,
        "m" => m,
        "nl_modes" => op.nl_modes,
        "matrix_size" => n
    )

    return A, B, interior_dofs, info
end

"""
    apply_sparse_boundary_conditions!(A, B, op)

Apply boundary conditions by replacing appropriate rows in A and B matrices.
Uses the tau method.

Mechanical BCs (controlled by bci/bco):
- 0 = stress-free: u = 0, ∂²u/∂r² = 0 (poloidal); ∂v/∂r = 0 (toroidal)
- 1 = no-slip: u = 0, ∂u/∂r = 0 (poloidal); v = 0 (toroidal)

Thermal BCs (controlled by bci_thermal/bco_thermal):
- 0 = fixed temperature: θ = 0
- 1 = fixed flux: ∂θ/∂r = 0
"""
function apply_sparse_boundary_conditions!(A::SparseMatrixCSC,
                                        B::SparseMatrixCSC,
                                        op::SparseStabilityOperator{T}) where {T}
    params = op.params
    N = params.N
    n_per_mode = N + 1

    nb_top = length(op.ll_top)
    nb_bot = length(op.ll_bot)

    # -------------------------------------------------------------------------
    # Poloidal velocity BCs
    # -------------------------------------------------------------------------
    for (k, l) in enumerate(op.ll_top)
        row_base = (k - 1) * n_per_mode

        # Outer boundary (r = ro = 1.0)
        if params.bco == 1
            # No-slip: u = 0, du/dr = 0
            apply_boundary_conditions!(A, B, [row_base + 1], :dirichlet, N,
                                      params.ricb, one(T))
            apply_boundary_conditions!(A, B, [row_base + 2], :neumann, N,
                                      params.ricb, one(T))
        else
            # Stress-free: u = 0, d²u/dr² = 0
            bc_rows = [row_base + 1, row_base + 2]
            apply_boundary_conditions!(A, B, [row_base + 1], :dirichlet, N,
                                      params.ricb, one(T))
            apply_boundary_conditions!(A, B, [row_base + 2], :neumann2, N,
                                      params.ricb, one(T))
        end

        # Inner boundary (r = ri = ricb)
        if params.bci == 1
            # No-slip: u = 0, du/dr = 0
            apply_boundary_conditions!(A, B, [row_base + n_per_mode], :dirichlet, N,
                                      params.ricb, one(T))
            apply_boundary_conditions!(A, B, [row_base + n_per_mode - 1], :neumann, N,
                                      params.ricb, one(T))
        else
            # Stress-free: u = 0, d²u/dr² = 0
            apply_boundary_conditions!(A, B, [row_base + n_per_mode], :dirichlet, N,
                                      params.ricb, one(T))
            apply_boundary_conditions!(A, B, [row_base + n_per_mode - 1], :neumann2, N,
                                      params.ricb, one(T))
        end
    end

    # -------------------------------------------------------------------------
    # Toroidal velocity BCs
    # -------------------------------------------------------------------------
    D1_toroidal = UltrasphericalSpectral.sparse_radial_operator(0, 1, N, params.ricb, one(T))
    outer_deriv_row = Vector(D1_toroidal[1, :])
    inner_deriv_row = Vector(D1_toroidal[N+1, :])

    for (k, l) in enumerate(op.ll_bot)
        row_base = (nb_top + k - 1) * n_per_mode

        # Outer boundary (r = ro = 1.0)
        if params.bco == 1
            # No-slip: v = 0
            apply_boundary_conditions!(A, B, [row_base + 1], :dirichlet, N,
                                      params.ricb, one(T))
        else
            # Stress-free: -r·∂v/∂r + v = 0
            row = row_base + 1
            A[row, :] .= 0.0
            B[row, :] .= 0.0
            block_start = row_base + 1
            for n in 0:N
                idx = block_start + n
                A[row, idx] = -outer_deriv_row[n+1]
                A[row, idx] += 1.0
            end
        end

        # Inner boundary (r = ri = ricb)
        if params.bci == 1
            # No-slip: v = 0
            apply_boundary_conditions!(A, B, [row_base + n_per_mode], :dirichlet, N,
                                      params.ricb, one(T))
        else
            # Stress-free: -r·∂v/∂r + v = 0
            row = row_base + n_per_mode
            A[row, :] .= 0.0
            B[row, :] .= 0.0
            block_start = row_base + 1
            for n in 0:N
                idx = block_start + n
                A[row, idx] = -params.ricb * inner_deriv_row[n+1]
                A[row, idx] += (-1.0)^n
            end
        end
    end

    # -------------------------------------------------------------------------
    # Temperature BCs
    # -------------------------------------------------------------------------
    for (k, l) in enumerate(op.ll_top)
        row_base = (nb_top + nb_bot + k - 1) * n_per_mode

        # Outer boundary (r = ro = 1.0)
        if params.bco_thermal == 0
            # Fixed temperature: θ = 0
            apply_boundary_conditions!(A, B, [row_base + 1], :dirichlet, N,
                                      params.ricb, one(T))
        else
            # Fixed flux: dθ/dr = 0
            apply_boundary_conditions!(A, B, [row_base + 1], :neumann, N,
                                      params.ricb, one(T))
        end

        # Inner boundary (r = ri = ricb)
        if params.bci_thermal == 0
            # Fixed temperature: θ = 0
            apply_boundary_conditions!(A, B, [row_base + n_per_mode], :dirichlet, N,
                                      params.ricb, one(T))
        else
            # Fixed flux: dθ/dr = 0
            apply_boundary_conditions!(A, B, [row_base + n_per_mode], :neumann, N,
                                      params.ricb, one(T))
        end
    end

    return nothing
end

end  # module SparseOperator
