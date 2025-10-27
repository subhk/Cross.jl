# =============================================================================
#  MHD Linear Stability Operator
#
#  Implementation of magnetohydrodynamic (MHD) operators for dynamo simulations
#  in rotating spherical shells following Kore's structure.
#
#  Extends the hydrodynamic operators with:
#  - Lorentz forces (magnetic field acting on velocity)
#  - Induction equation (velocity advecting magnetic field)
#  - Magnetic diffusion
#
#  References:
#  - Kore: kore-main/bin/operators.py, submatrices.py, assemble.py
#  - Christensen & Wicht (2015), "Numerical Dynamo Simulations"
#  - Jones (2011), "Planetary Magnetic Fields and Fluid Dynamos"
# =============================================================================

module MHDOperator

using LinearAlgebra
using SparseArrays
using Printf
using SpecialFunctions

# Import from other modules
push!(LOAD_PATH, @__DIR__)
include("UltrasphericalSpectral.jl")
using .UltrasphericalSpectral

include("DipoleOperators.jl")

export MHDParams,
       MHDStabilityOperator,
       assemble_mhd_matrices,
       BackgroundField,
       no_field, axial, dipole,
       is_dipole_case,
       radial_power_shift_poloidal,
       radial_power_shift_toroidal,
       radial_power_shift_magnetic_poloidal,
       radial_power_shift_magnetic_toroidal

# -----------------------------------------------------------------------------
# Background magnetic field types
# -----------------------------------------------------------------------------

"""
Background magnetic field types supported by the code.

Options:
- `:none` - No background field (kinematic dynamo)
- `:axial` - Uniform axial field B₀ = B₀ẑ
- `:dipole` - Dipolar field B₀ ~ (2cosθ r̂ + sinθ θ̂)/r³
"""
@enum BackgroundField begin
    no_field = 0
    axial = 1
    dipole = 2
end

# -----------------------------------------------------------------------------
# MHD Parameters
# -----------------------------------------------------------------------------

"""
    MHDParams{T<:Real}

Parameters for magnetohydrodynamic (MHD) dynamo simulations in rotating spherical shells.

This structure contains all physical, geometric, and numerical parameters needed for
MHD linear stability analysis or dynamo onset calculations.

# Physical Parameters (Dimensionless)

- `E::T`: **Ekman number** = ν/(ΩL²)
  - Ratio of viscous to Coriolis forces
  - Typical values: 10⁻³ (lab) to 10⁻¹⁵ (Earth's core)

- `Pr::T`: **Prandtl number** = ν/κ
  - Ratio of momentum to thermal diffusivity
  - Typical values: 0.1-10 for liquid metals, ~1 for Earth's core

- `Pm::T`: **Magnetic Prandtl number** = ν/η
  - Ratio of momentum to magnetic diffusivity
  - Earth's core: Pm ~ 10⁻⁶, Lab experiments: Pm ~ 10⁻⁵

- `Ra::T`: **Rayleigh number** = αgΔTL³/(νκ)
  - Measure of buoyancy forcing strength
  - Critical value Raᶜ determines onset of convection

- `Le::T`: **Lehnert number** = B₀/(√(μρ)ΩL)
  - Measure of background magnetic field strength
  - Le = 0: Pure hydrodynamic case
  - Typical values: 10⁻⁴ to 10⁻² for planetary dynamos

# Geometry

- `ricb::T`: **Inner core radius** (normalized, outer radius = 1)
  - Must satisfy 0 < ricb < 1
  - Earth's core: χ ≈ 0.35
  - ricb = 0: Full sphere (no inner core)

- `m::Int`: **Azimuthal wavenumber**
  - Fourier mode number in longitude (φ-direction)
  - m = 0: Axisymmetric
  - Typical range: m = 0 to ~10 for onset studies

- `lmax::Int`: **Maximum spherical harmonic degree**
  - Controls spectral truncation in θ-direction
  - Must satisfy lmax ≥ m
  - Typical values: 10-50 (onset), 100+ (turbulence)

- `symm::Int`: **Equatorial symmetry**
  - symm = +1: Equatorially symmetric modes
  - symm = -1: Equatorially antisymmetric modes
  - Affects mode parity selection

- `N::Int`: **Number of radial collocation points**
  - Must be even and ≥ 4
  - Determines radial resolution
  - Typical values: 24-64 for onset, 128+ for turbulence

# Background Magnetic Field

- `B0_type::BackgroundField`: Type of imposed field
  - `no_field`: Pure hydrodynamic (kinematic dynamo)
  - `axial`: Uniform B₀ = B₀ẑ (simplest MHD case)
  - `dipole`: Dipolar field (requires ricb > 0)

- `B0_amplitude::T`: Dimensionless field strength
  - Scales the background field
  - Related to Lehnert number: Le ~ B0_amplitude

# Boundary Conditions

## Mechanical (Velocity)
- `bci::Int`: Inner core mechanical BC
  - 0 = **stress-free**: u = 0, ∂²u/∂r² = 0 (poloidal); ∂v/∂r = 0 (toroidal)
  - 1 = **no-slip**: u = 0, ∂u/∂r = 0 (poloidal); v = 0 (toroidal)

- `bco::Int`: Outer boundary (CMB) mechanical BC
  - Same options as bci
  - Most common: no-slip at both boundaries (bci=bco=1)

## Thermal
- `bci_thermal::Int`: Inner core thermal BC
  - 0 = **fixed temperature**: T = 0
  - 1 = **fixed flux**: ∂T/∂r = 0

- `bco_thermal::Int`: Outer boundary (CMB) thermal BC
  - Same options as bci_thermal
  - Common choice: fixed temperature at both (bci_thermal=bco_thermal=0)

## Magnetic
- `bci_magnetic::Int`: Inner core magnetic BC
  - 0 = **insulating**: B·n̂ continuous, no currents
  - 1 = **conducting** (finite conductivity): uses Bessel functions
  - 2 = **perfect conductor**: tangential E = 0

- `bco_magnetic::Int`: Outer boundary (CMB) magnetic BC
  - 0 = **insulating**: Most common (electrically insulating mantle)
  - 1,2 = conducting/perfect (rarely used at CMB)

# Heating Mode

- `heating::Symbol`: Thermal forcing configuration
  - `:differential`: Bottom heated, top cooled (classical Rayleigh-Bénard)
  - `:internal`: Volumetric heat sources (radioactive heating)

# Derived Quantities (Automatically Computed)

- `L::T`: **Shell thickness** = 1 - ricb
- `Etherm::T`: **Thermal Ekman number** = E/Pr = κ/(ΩL²)
- `Em::T`: **Magnetic Ekman number** = E/Pm = η/(ΩL²)

# Examples

```julia
# Basic hydrodynamic onset (Christensen & Wicht 2015, Table 1)
params_hydro = MHDParams(
    E=4.734e-5, Pr=1.0, Pm=1.0, Ra=1.6e6, Le=0.0,
    ricb=0.35, m=9, lmax=20, N=24,
    bci=1, bco=1,  # no-slip boundaries
    bci_thermal=0, bco_thermal=0,  # fixed temperature
    bci_magnetic=0, bco_magnetic=0  # insulating (irrelevant for Le=0)
)

# MHD dynamo with axial background field
params_mhd = MHDParams(
    E=1e-3, Pr=1.0, Pm=5.0, Ra=1e5, Le=1e-3,
    ricb=0.35, m=2, lmax=15, N=32,
    B0_type=axial,
    bci=1, bco=1,
    bci_thermal=0, bco_thermal=0,
    bci_magnetic=0, bco_magnetic=0  # insulating boundaries
)

# Perfect conductor inner core
params_conductor = MHDParams(
    E=1e-3, Pr=1.0, Pm=5.0, Ra=1e5, Le=1e-3,
    ricb=0.35, m=2, lmax=15, N=32,
    B0_type=axial,
    bci=1, bco=1,
    bci_thermal=0, bco_thermal=0,
    bci_magnetic=2, bco_magnetic=0  # perfect conductor ICB
)
```

# References

- Christensen & Wicht (2015), "Numerical Dynamo Simulations", Treatise on Geophysics
- Jones (2011), "Planetary Magnetic Fields and Fluid Dynamos", Ann. Rev. Fluid Mech.
- Kore implementation: Rekier et al. (2019)

# See Also

- [`MHDStabilityOperator`](@ref): Build operators from these parameters
- [`assemble_mhd_matrices`](@ref): Assemble eigenvalue problem matrices
"""
struct MHDParams{T<:Real}
    # Physical parameters
    E::T
    Pr::T
    Pm::T
    Ra::T
    Le::T

    # Geometry
    ricb::T
    m::Int
    lmax::Int
    symm::Int
    N::Int

    # Background field
    B0_type::BackgroundField
    B0_amplitude::T

    # Boundary conditions
    bci::Int
    bco::Int
    bci_thermal::Int
    bco_thermal::Int
    bci_magnetic::Int
    bco_magnetic::Int

    # Heating
    heating::Symbol

    # Derived
    L::T
    Etherm::T
    Em::T

    function MHDParams{T}(E, Pr, Pm, Ra, Le, ricb, m, lmax, symm, N,
                         B0_type, B0_amplitude,
                         bci, bco, bci_thermal, bco_thermal,
                         bci_magnetic, bco_magnetic,
                         heating, L, Etherm, Em) where {T<:Real}
        @assert 0 < ricb < 1 "ricb must be in (0,1)"
        @assert E > 0 "Ekman number must be positive"
        @assert Pr > 0 "Prandtl number must be positive"
        @assert Pm > 0 "Magnetic Prandtl number must be positive"
        @assert Ra > 0 "Rayleigh number must be positive"
        @assert lmax >= m "lmax must be >= m"
        @assert N >= 4 && iseven(N) "N must be even and >= 4"
        @assert symm in (-1, 1) "symm must be ±1"
        @assert heating in (:internal, :differential) "heating must be :internal or :differential"

        new{T}(E, Pr, Pm, Ra, Le, ricb, m, lmax, symm, N,
               B0_type, B0_amplitude,
               bci, bco, bci_thermal, bco_thermal,
               bci_magnetic, bco_magnetic,
               heating, L, Etherm, Em)
    end
end

# Constructor with keyword arguments
function MHDParams(; E, Pr=1.0, Pm=1.0, Ra, ricb,
                   Le=0.0,
                   m::Int, lmax::Int, symm::Int=1, N::Int,
                   B0_type::BackgroundField=no_field,
                   B0_amplitude=0.0,
                   bci::Int=1, bco::Int=1,
                   bci_thermal::Int=0, bco_thermal::Int=0,
                   bci_magnetic::Int=0, bco_magnetic::Int=0,
                   heating::Symbol=:differential)
    # Promote all numeric parameters to common type
    T = promote_type(typeof(E), typeof(Pr), typeof(Pm), typeof(Ra), typeof(ricb), typeof(Le), typeof(B0_amplitude))
    E_T = T(E)
    Pr_T = T(Pr)
    Pm_T = T(Pm)
    Ra_T = T(Ra)
    ricb_T = T(ricb)
    Le_T = T(Le)
    B0_amplitude_T = T(B0_amplitude)

    L = one(T) - ricb_T
    Etherm = E_T / Pr_T
    Em = E_T / Pm_T
    return MHDParams{T}(E_T, Pr_T, Pm_T, Ra_T, Le_T, ricb_T, m, lmax, symm, N,
                       B0_type, B0_amplitude_T,
                       bci, bco, bci_thermal, bco_thermal,
                       bci_magnetic, bco_magnetic,
                       heating, L, Etherm, Em)
end

# -----------------------------------------------------------------------------
# MHD Stability Operator Structure
# -----------------------------------------------------------------------------

"""
    MHDStabilityOperator

Stores pre-computed sparse radial operators for MHD stability analysis.

Extends the hydrodynamic operators with magnetic field operators.

# Sections
- u: Poloidal velocity (2curl Navier-Stokes)
- v: Toroidal velocity (1curl Navier-Stokes)
- f: Poloidal magnetic field (no-curl induction)
- g: Toroidal magnetic field (1curl induction)
- h: Temperature perturbation
"""
struct MHDStabilityOperator{T<:Real}
    params::MHDParams{T}

    # Velocity operators (same as hydrodynamic case)
    r0_D0_u::SparseMatrixCSC{Float64,Int}
    r2_D0_u::SparseMatrixCSC{Float64,Int}
    r2_D2_u::SparseMatrixCSC{Float64,Int}
    r3_D0_u::SparseMatrixCSC{Float64,Int}
    r3_D1_u::SparseMatrixCSC{Float64,Int}
    r4_D0_u::SparseMatrixCSC{Float64,Int}
    r4_D1_u::SparseMatrixCSC{Float64,Int}
    r4_D2_u::SparseMatrixCSC{Float64,Int}
    r3_D3_u::SparseMatrixCSC{Float64,Int}
    r4_D4_u::SparseMatrixCSC{Float64,Int}

    r0_D0_v::SparseMatrixCSC{Float64,Int}
    r1_D0_v::SparseMatrixCSC{Float64,Int}
    r1_D1_v::SparseMatrixCSC{Float64,Int}
    r2_D0_v::SparseMatrixCSC{Float64,Int}
    r2_D1_v::SparseMatrixCSC{Float64,Int}
    r2_D2_v::SparseMatrixCSC{Float64,Int}

    # Magnetic field operators for poloidal field (f)
    r0_D0_f::SparseMatrixCSC{Float64,Int}
    r1_D0_f::SparseMatrixCSC{Float64,Int}
    r1_D1_f::SparseMatrixCSC{Float64,Int}
    r2_D0_f::SparseMatrixCSC{Float64,Int}
    r2_D1_f::SparseMatrixCSC{Float64,Int}
    r2_D2_f::SparseMatrixCSC{Float64,Int}

    # Magnetic field operators for toroidal field (g)
    r0_D0_g::SparseMatrixCSC{Float64,Int}
    r1_D0_g::SparseMatrixCSC{Float64,Int}
    r1_D1_g::SparseMatrixCSC{Float64,Int}
    r2_D0_g::SparseMatrixCSC{Float64,Int}
    r2_D1_g::SparseMatrixCSC{Float64,Int}
    r2_D2_g::SparseMatrixCSC{Float64,Int}

    # Temperature operators
    r0_D0_h::SparseMatrixCSC{Float64,Int}
    r1_D0_h::SparseMatrixCSC{Float64,Int}
    r1_D1_h::SparseMatrixCSC{Float64,Int}
    r2_D0_h::SparseMatrixCSC{Float64,Int}
    r2_D1_h::SparseMatrixCSC{Float64,Int}
    r2_D2_h::SparseMatrixCSC{Float64,Int}
    r3_D0_h::SparseMatrixCSC{Float64,Int}
    r3_D2_h::SparseMatrixCSC{Float64,Int}

    # Background field operators (for Lorentz force and induction)
    # h(r) represents the background magnetic field structure function
    r0_h0_D0::SparseMatrixCSC{Float64,Int}  # For axial: h(r) = r
    r1_h0_D0::SparseMatrixCSC{Float64,Int}
    r1_h0_D1::SparseMatrixCSC{Float64,Int}
    r2_h0_D0::SparseMatrixCSC{Float64,Int}
    r2_h0_D1::SparseMatrixCSC{Float64,Int}
    r2_h0_D2::SparseMatrixCSC{Float64,Int}

    # Mode structure
    ll_u::Vector{Int}  # l-modes for poloidal velocity
    ll_v::Vector{Int}  # l-modes for toroidal velocity
    ll_f::Vector{Int}  # l-modes for poloidal magnetic field
    ll_g::Vector{Int}  # l-modes for toroidal magnetic field
    ll_h::Vector{Int}  # l-modes for temperature

    nl_modes::Int
    matrix_size::Int
end

# Constructor
function MHDStabilityOperator(params::MHDParams{T}) where {T}
    println("Building MHD sparse operators (N=$(params.N), ricb=$(params.ricb))...")

    N = params.N
    ri = params.ricb
    ro = one(T)

    # Compute all radial operators
    println("  Computing velocity operators...")
    r0_D0_u = sparse_radial_operator(0, 0, N, ri, ro)
    r2_D0_u = sparse_radial_operator(2, 0, N, ri, ro)
    r2_D2_u = sparse_radial_operator(2, 2, N, ri, ro)
    r3_D0_u = sparse_radial_operator(3, 0, N, ri, ro)
    r3_D1_u = sparse_radial_operator(3, 1, N, ri, ro)
    r4_D0_u = sparse_radial_operator(4, 0, N, ri, ro)
    r4_D1_u = sparse_radial_operator(4, 1, N, ri, ro)
    r4_D2_u = sparse_radial_operator(4, 2, N, ri, ro)
    r3_D3_u = sparse_radial_operator(3, 3, N, ri, ro)
    r4_D4_u = sparse_radial_operator(4, 4, N, ri, ro)

    r0_D0_v = sparse_radial_operator(0, 0, N, ri, ro)
    r1_D0_v = sparse_radial_operator(1, 0, N, ri, ro)
    r1_D1_v = sparse_radial_operator(1, 1, N, ri, ro)
    r2_D0_v = sparse_radial_operator(2, 0, N, ri, ro)
    r2_D1_v = sparse_radial_operator(2, 1, N, ri, ro)
    r2_D2_v = sparse_radial_operator(2, 2, N, ri, ro)

    println("  Computing magnetic field operators...")
    r0_D0_f = sparse_radial_operator(0, 0, N, ri, ro)
    r1_D0_f = sparse_radial_operator(1, 0, N, ri, ro)
    r1_D1_f = sparse_radial_operator(1, 1, N, ri, ro)
    r2_D0_f = sparse_radial_operator(2, 0, N, ri, ro)
    r2_D1_f = sparse_radial_operator(2, 1, N, ri, ro)
    r2_D2_f = sparse_radial_operator(2, 2, N, ri, ro)

    r0_D0_g = sparse_radial_operator(0, 0, N, ri, ro)
    r1_D0_g = sparse_radial_operator(1, 0, N, ri, ro)
    r1_D1_g = sparse_radial_operator(1, 1, N, ri, ro)
    r2_D0_g = sparse_radial_operator(2, 0, N, ri, ro)
    r2_D1_g = sparse_radial_operator(2, 1, N, ri, ro)
    r2_D2_g = sparse_radial_operator(2, 2, N, ri, ro)

    println("  Computing temperature operators...")
    r0_D0_h = sparse_radial_operator(0, 0, N, ri, ro)
    r1_D0_h = sparse_radial_operator(1, 0, N, ri, ro)
    r1_D1_h = sparse_radial_operator(1, 1, N, ri, ro)
    r2_D0_h = sparse_radial_operator(2, 0, N, ri, ro)
    r2_D1_h = sparse_radial_operator(2, 1, N, ri, ro)
    r2_D2_h = sparse_radial_operator(2, 2, N, ri, ro)
    r3_D0_h = sparse_radial_operator(3, 0, N, ri, ro)
    r3_D2_h = sparse_radial_operator(3, 2, N, ri, ro)

    println("  Computing background field operators...")
    # For axial field: h(r) = r, so these are just r^k operators
    # For dipole field: h(r) would be more complex (implemented separately)
    r0_h0_D0 = sparse_radial_operator(1, 0, N, ri, ro)  # r^1 * D^0
    r1_h0_D0 = sparse_radial_operator(2, 0, N, ri, ro)  # r^2 * D^0
    r1_h0_D1 = sparse_radial_operator(2, 1, N, ri, ro)  # r^2 * D^1
    r2_h0_D0 = sparse_radial_operator(3, 0, N, ri, ro)  # r^3 * D^0
    r2_h0_D1 = sparse_radial_operator(3, 1, N, ri, ro)  # r^3 * D^1
    r2_h0_D2 = sparse_radial_operator(3, 2, N, ri, ro)  # r^3 * D^2

    # Determine l-mode structure
    ll_u, ll_v = compute_mhd_l_modes(params.m, params.lmax, params.symm, params.B0_type)
    ll_f = ll_u  # Poloidal magnetic field has same parity as poloidal velocity
    ll_g = ll_v  # Toroidal magnetic field has same parity as toroidal velocity
    ll_h = ll_u  # Temperature has same parity as poloidal velocity

    nl_modes = length(ll_u) + length(ll_v)
    n_per_mode = N + 1

    # Matrix size: 4 sections (u, v, f, g, h)
    # For MHD: u, v, f, g, h
    matrix_size = 2 * nl_modes * n_per_mode +  # u, v
                  2 * nl_modes * n_per_mode +  # f, g
                  length(ll_h) * n_per_mode     # h

    println("  l-modes: $(length(ll_u)) poloidal + $(length(ll_v)) toroidal")
    println("  Matrix size: $(matrix_size) × $(matrix_size)")
    println("  Estimated sparsity: ~$(estimate_mhd_sparsity(N, nl_modes))%")

    return MHDStabilityOperator{T}(
        params,
        r0_D0_u, r2_D0_u, r2_D2_u, r3_D0_u, r3_D1_u, r4_D0_u, r4_D1_u, r4_D2_u, r3_D3_u, r4_D4_u,
        r0_D0_v, r1_D0_v, r1_D1_v, r2_D0_v, r2_D1_v, r2_D2_v,
        r0_D0_f, r1_D0_f, r1_D1_f, r2_D0_f, r2_D1_f, r2_D2_f,
        r0_D0_g, r1_D0_g, r1_D1_g, r2_D0_g, r2_D1_g, r2_D2_g,
        r0_D0_h, r1_D0_h, r1_D1_h, r2_D0_h, r2_D1_h, r2_D2_h, r3_D0_h, r3_D2_h,
        r0_h0_D0, r1_h0_D0, r1_h0_D1, r2_h0_D0, r2_h0_D1, r2_h0_D2,
        ll_u, ll_v, ll_f, ll_g, ll_h,
        nl_modes, matrix_size
    )
end

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

"""
    compute_mhd_l_modes(m, lmax, symm, B0_type)

Compute l-mode indices for MHD problem.
The magnetic field symmetry depends on the background field type.
"""
function compute_mhd_l_modes(m::Int, lmax::Int, symm::Int, B0_type::BackgroundField)
    if symm == 1
        # Equatorially symmetric flow
        ll_u = collect(m:2:lmax)      # Poloidal velocity
        ll_v = collect((m+1):2:lmax)  # Toroidal velocity
    elseif symm == -1
        # Equatorially antisymmetric flow
        ll_u = collect((m+1):2:lmax)
        ll_v = collect(m:2:lmax)
    else
        # Both symmetries
        ll_u = collect(m:lmax)
        ll_v = collect(m:lmax)
    end

    return ll_u, ll_v
end

function estimate_mhd_sparsity(N::Int, nl_modes::Int)
    # MHD has more couplings than pure hydro
    # Each field couples to multiple others via Lorentz force and induction
    total_size = 5 * nl_modes * (N + 1)  # u, v, f, g, h
    nnz_estimate = 10 * nl_modes * N^2   # More couplings than hydro
    sparsity = 100.0 * (1.0 - nnz_estimate / total_size^2)
    return round(sparsity, digits=2)
end

# Continued in next part...

end  # module MHDOperator
