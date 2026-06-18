# =============================================================================
#  Basic State for Onset of Convection
#
#  Implements both axisymmetric and non-axisymmetric basic states with:
#  - Temperature variations: θ̄(r,θ,φ)
#  - Thermal wind-balanced flows: ū(r,θ,φ)
#
#  Two implementations:
#  1. BasicState: Axisymmetric (m=0 only), for standard onset
#  2. BasicState3D: Non-axisymmetric (multiple m), for tri-global analysis
# =============================================================================

using Parameters
using LinearAlgebra
using SparseArrays

"""
    BasicState{T<:Real}

Holds the axisymmetric (m=0) basic state for linear stability analysis.

The basic state consists of:
- Temperature: θ̄(r,θ) = Σ_ℓ θ̄_ℓ0(r) Y_ℓ0(θ)
- Zonal flow: ū_φ(r,θ) = Σ_ℓ ū_φ,ℓ0(r) Y_ℓ0(θ)
- No meridional flow: ū_r = ū_θ = 0

Fields:
- `lmax_bs::Int` - Maximum spherical harmonic degree for basic state
- `Nr::Int` - Number of radial collocation points
- `r::Vector{T}` - Radial collocation points
- `theta_coeffs::Dict{Int,Vector{T}}` - Temperature coefficients θ̄_ℓ0(r) for each ℓ
- `uphi_coeffs::Dict{Int,Vector{T}}` - Zonal flow coefficients ū_φ,ℓ0(r) for each ℓ
- `dtheta_dr_coeffs::Dict{Int,Vector{T}}` - Radial derivative ∂θ̄_ℓ0/∂r
- `duphi_dr_coeffs::Dict{Int,Vector{T}}` - Radial derivative ∂ū_φ,ℓ0/∂r
"""
@with_kw_noshow struct BasicState{T<:Real}
    lmax_bs::Int
    Nr::Int
    r::Vector{T}
    theta_coeffs::Dict{Int,Vector{T}}
    uphi_coeffs::Dict{Int,Vector{T}}
    dtheta_dr_coeffs::Dict{Int,Vector{T}}
    duphi_dr_coeffs::Dict{Int,Vector{T}}
end


"""
    BasicState3D{T<:Real}

Holds a non-axisymmetric (3D) basic state for tri-global instability analysis.

The basic state has both meridional AND longitudinal variations:
- Temperature: θ̄(r,θ,φ) = Σ_ℓ Σ_m_bs θ̄_ℓm_bs(r) Y_ℓm_bs(θ,φ)
- Velocity components: ū_r, ū_θ, ū_φ from thermal-wind/geostrophic balance

This enables studying onset of convection on top of 3D thermal and flow structures,
such as:
- Longitudinally-varying boundary heating
- Zonal jets with wavenumber structure
- Realistic 3D planetary/stellar base states

Fields:
- `lmax_bs::Int` - Maximum spherical harmonic degree
- `mmax_bs::Int` - Maximum azimuthal wavenumber (typically small, e.g., 0-4)
- `Nr::Int` - Number of radial collocation points
- `r::Vector{T}` - Radial collocation points
- `theta_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - θ̄_ℓm(r) indexed by (ℓ,m)
- `ur_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ū_r,ℓm(r)
- `utheta_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ū_θ,ℓm(r)
- `uphi_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ū_φ,ℓm(r)
- `dtheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ∂θ̄_ℓm/∂r
- `dur_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ∂ū_r,ℓm/∂r
- `dutheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ∂ū_θ,ℓm/∂r
- `duphi_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ∂ū_φ,ℓm/∂r

Note: Perturbations on this basic state couple multiple azimuthal modes m simultaneously.
The eigenvalue problem becomes block-coupled across different m values.
"""
@with_kw_noshow struct BasicState3D{T<:Real}
    lmax_bs::Int
    mmax_bs::Int
    Nr::Int
    r::Vector{T}
    # Temperature
    theta_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    dtheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}

    # Velocity components
    ur_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    utheta_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    uphi_coeffs::Dict{Tuple{Int,Int},Vector{T}}

    # Velocity derivatives
    dur_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    dutheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    duphi_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
end


# =============================================================================
#  Symbolic Spherical Harmonic Boundary Conditions
#
#  Provides an intuitive interface for specifying temperature boundary
#  conditions using spherical harmonic notation:
#
#    bc = Y20(0.1) + Y22(0.05)  # Meridional + longitudinal pattern
#    bc = 0.5 * Y10()           # Scaled dipole
#
#  These can be passed directly to basic state functions.
# =============================================================================

"""
    SphericalHarmonicBC{T<:Real}

Symbolic representation of boundary conditions expanded in spherical harmonics.

This type provides a convenient way to specify temperature boundary conditions
using standard spherical harmonic notation (Y_ℓm) rather than dictionary syntax.

# Constructor Functions
- `Ylm(ℓ, m, amplitude)` - General spherical harmonic mode
- `Y00(amp)`, `Y10(amp)`, `Y11(amp)` - Monopole and dipole
- `Y20(amp)`, `Y21(amp)`, `Y22(amp)` - Quadrupole
- `Y30(amp)`, ..., `Y44(amp)` - Higher orders

# Operators
- `+` : Combine multiple harmonics: `Y20(0.1) + Y22(0.05)`
- `*` : Scale amplitude: `0.5 * Y20(0.1)` or `Y20(0.1) * 0.5`
- `-` : Negate or subtract: `-Y20(0.1)` or `Y20(0.1) - Y22(0.05)`

# Examples

## Simple meridional variation (Y₂₀)
```julia
bc = Y20(0.1)
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)
```

## Combined meridional and longitudinal variation
```julia
bc = Y20(0.1) + Y22(0.05)
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)
```

## Dipole pattern with negative amplitude
```julia
bc = Y10(-0.2)  # Hot at one pole, cold at other
```

## Complex pattern with scaling
```julia
bc = 0.5 * (Y20(1.0) + 2.0 * Y40(0.5))
```

## Flux boundary condition
```julia
bc = Y20(0.1)  # Interpreted as flux when thermal_bc=:fixed_flux
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc, thermal_bc=:fixed_flux)
```

# Physical Interpretation

The spherical harmonics Y_ℓm represent different angular patterns:

- **Y₀₀**: Uniform (spherically symmetric)
- **Y₁₀**: North-south dipole (one hemisphere hot, other cold)
- **Y₁₁**: East-west dipole
- **Y₂₀**: Equator-pole contrast (equator different from poles)
- **Y₂₂**: Four-fold longitudinal pattern (warm at 0°,180°, cold at 90°,270°)
- **Y₃₀**: More complex latitudinal structure
- **Y₄₀**: Even more latitudinal bands

For convection studies:
- Y₂₀ is most common: represents differential heating between equator and poles
- Y₂₂ represents tidal or orbital forcing patterns
- Y₁₀ represents hemispherical asymmetry
"""
struct SphericalHarmonicBC{T<:Real}
    coeffs::Dict{Tuple{Int,Int}, T}
end

# Empty constructor
"""Create an empty symbolic spherical-harmonic boundary condition."""
SphericalHarmonicBC{T}() where T = SphericalHarmonicBC{T}(Dict{Tuple{Int,Int}, T}())

# Single mode constructor
"""Create a symbolic boundary condition containing one `(l, m)` harmonic."""
function SphericalHarmonicBC(ℓ::Int, m::Int, amplitude::T) where T<:Real
    if ℓ < 0
        throw(ArgumentError("ℓ must be non-negative, got ℓ=$ℓ"))
    end
    if m < 0 || m > ℓ
        throw(ArgumentError("m must satisfy 0 ≤ m ≤ ℓ, got ℓ=$ℓ, m=$m"))
    end
    SphericalHarmonicBC{T}(Dict((ℓ, m) => amplitude))
end

# Addition: combine multiple spherical harmonic BCs
"""Combine two symbolic boundary-condition spectra, promoting amplitudes as needed."""
function Base.:+(a::SphericalHarmonicBC{T}, b::SphericalHarmonicBC{S}) where {T,S}
    R = promote_type(T, S)
    result = Dict{Tuple{Int,Int}, R}()
    for (k, v) in a.coeffs
        result[k] = get(result, k, zero(R)) + R(v)
    end
    for (k, v) in b.coeffs
        result[k] = get(result, k, zero(R)) + R(v)
    end
    SphericalHarmonicBC{R}(result)
end

# Scalar multiplication (from left)
"""Scale all amplitudes in a symbolic spherical-harmonic boundary condition."""
function Base.:*(c::Real, bc::SphericalHarmonicBC{T}) where T
    R = promote_type(typeof(c), T)
    SphericalHarmonicBC{R}(Dict(k => R(c) * R(v) for (k, v) in bc.coeffs))
end

# Scalar multiplication (from right)
"""Scale a symbolic spherical-harmonic boundary condition from the right."""
Base.:*(bc::SphericalHarmonicBC, c::Real) = c * bc

# Division by scalar
"""Divide all amplitudes in a symbolic boundary condition by a scalar."""
Base.:/(bc::SphericalHarmonicBC, c::Real) = (1/c) * bc

# Negation
"""Negate all amplitudes in a symbolic boundary condition."""
Base.:-(bc::SphericalHarmonicBC) = (-1) * bc

# Subtraction
"""Subtract one symbolic spherical-harmonic boundary condition from another."""
Base.:-(a::SphericalHarmonicBC, b::SphericalHarmonicBC) = a + (-b)

# Zero check
"""Return true when all stored boundary-condition amplitudes are zero."""
Base.iszero(bc::SphericalHarmonicBC) = isempty(bc.coeffs) || all(iszero, values(bc.coeffs))

# =============================================================================
#  Convenience Constructors for Common Spherical Harmonics
# =============================================================================

"""
    Ylm(ℓ::Int, m::Int, amplitude::Real=1.0)

Create a spherical harmonic boundary condition for mode (ℓ, m).

# Arguments
- `ℓ` : Spherical harmonic degree (ℓ ≥ 0)
- `m` : Azimuthal order (0 ≤ m ≤ ℓ)
- `amplitude` : Amplitude of this mode (default: 1.0)

# Example
```julia
bc = Ylm(3, 2, 0.1)  # Y₃₂ mode with amplitude 0.1
```
"""
Ylm(ℓ::Int, m::Int, amplitude::Real=1.0) = SphericalHarmonicBC(ℓ, m, amplitude)

# ℓ = 0: Monopole (uniform)
"""Y00(amplitude=1.0) - Uniform (monopole) spherical harmonic"""
Y00(amplitude::Real=1.0) = SphericalHarmonicBC(0, 0, amplitude)

# ℓ = 1: Dipole
"""Y10(amplitude=1.0) - Axial dipole: cos(θ) pattern (north-south asymmetry)"""
Y10(amplitude::Real=1.0) = SphericalHarmonicBC(1, 0, amplitude)

"""Y11(amplitude=1.0) - Equatorial dipole: sin(θ)cos(φ) pattern"""
Y11(amplitude::Real=1.0) = SphericalHarmonicBC(1, 1, amplitude)

# ℓ = 2: Quadrupole
"""Y20(amplitude=1.0) - Axisymmetric quadrupole: (3cos²θ - 1) pattern (equator-pole contrast)"""
Y20(amplitude::Real=1.0) = SphericalHarmonicBC(2, 0, amplitude)

"""Y21(amplitude=1.0) - Tesseral quadrupole: sin(θ)cos(θ)cos(φ) pattern"""
Y21(amplitude::Real=1.0) = SphericalHarmonicBC(2, 1, amplitude)

"""Y22(amplitude=1.0) - Sectoral quadrupole: sin²(θ)cos(2φ) pattern (four-fold longitudinal)"""
Y22(amplitude::Real=1.0) = SphericalHarmonicBC(2, 2, amplitude)

# ℓ = 3: Octupole
"""Y30(amplitude=1.0) - Axisymmetric octupole"""
Y30(amplitude::Real=1.0) = SphericalHarmonicBC(3, 0, amplitude)

"""Y31(amplitude=1.0) - Tesseral octupole m=1"""
Y31(amplitude::Real=1.0) = SphericalHarmonicBC(3, 1, amplitude)

"""Y32(amplitude=1.0) - Tesseral octupole m=2"""
Y32(amplitude::Real=1.0) = SphericalHarmonicBC(3, 2, amplitude)

"""Y33(amplitude=1.0) - Sectoral octupole"""
Y33(amplitude::Real=1.0) = SphericalHarmonicBC(3, 3, amplitude)

# ℓ = 4: Hexadecapole
"""Y40(amplitude=1.0) - Axisymmetric hexadecapole"""
Y40(amplitude::Real=1.0) = SphericalHarmonicBC(4, 0, amplitude)

"""Y41(amplitude=1.0) - Tesseral hexadecapole m=1"""
Y41(amplitude::Real=1.0) = SphericalHarmonicBC(4, 1, amplitude)

"""Y42(amplitude=1.0) - Tesseral hexadecapole m=2"""
Y42(amplitude::Real=1.0) = SphericalHarmonicBC(4, 2, amplitude)

"""Y43(amplitude=1.0) - Tesseral hexadecapole m=3"""
Y43(amplitude::Real=1.0) = SphericalHarmonicBC(4, 3, amplitude)

"""Y44(amplitude=1.0) - Sectoral hexadecapole"""
Y44(amplitude::Real=1.0) = SphericalHarmonicBC(4, 4, amplitude)

# =============================================================================
#  Utility Functions for SphericalHarmonicBC
# =============================================================================

"""
    to_dict(bc::SphericalHarmonicBC)

Convert SphericalHarmonicBC to Dict{Tuple{Int,Int}, T} format.

This is the internal format used by basic state functions.
"""
to_dict(bc::SphericalHarmonicBC{T}) where T = Dict{Tuple{Int,Int}, T}(bc.coeffs)

"""
    get_lmax(bc::SphericalHarmonicBC)

Get the maximum spherical harmonic degree in the boundary condition.
"""
function get_lmax(bc::SphericalHarmonicBC)
    isempty(bc.coeffs) && return 0
    maximum(first(k) for k in keys(bc.coeffs))
end

"""
    get_mmax(bc::SphericalHarmonicBC)

Get the maximum azimuthal order in the boundary condition.
"""
function get_mmax(bc::SphericalHarmonicBC)
    isempty(bc.coeffs) && return 0
    maximum(last(k) for k in keys(bc.coeffs))
end

"""
    get_lmax_mmax(bc::SphericalHarmonicBC)

Get both (lmax, mmax) from the boundary condition.
"""
get_lmax_mmax(bc::SphericalHarmonicBC) = (get_lmax(bc), get_mmax(bc))

"""
    is_axisymmetric(bc::SphericalHarmonicBC)

Check if the boundary condition is axisymmetric (m=0 only).
"""
is_axisymmetric(bc::SphericalHarmonicBC) = get_mmax(bc) == 0

# Pretty printing
"""Print a compact algebraic representation of a spherical-harmonic boundary condition."""
function Base.show(io::IO, bc::SphericalHarmonicBC{T}) where T
    if isempty(bc.coeffs)
        print(io, "SphericalHarmonicBC{$T}(empty)")
        return
    end

    terms = String[]
    for ((ℓ, m), amp) in sort(collect(bc.coeffs), by=x->(x[1][1], x[1][2]))
        if abs(amp) < eps(T) * 1000
            continue
        end
        if amp == 1.0
            push!(terms, "Y$ℓ$m")
        elseif amp == -1.0
            push!(terms, "-Y$ℓ$m")
        else
            push!(terms, "$(amp)*Y$ℓ$m")
        end
    end

    if isempty(terms)
        print(io, "SphericalHarmonicBC{$T}(zero)")
    else
        print(io, join(terms, " + "))
    end
end

"""Print all stored harmonic amplitudes in a multiline REPL summary."""
function Base.show(io::IO, ::MIME"text/plain", bc::SphericalHarmonicBC{T}) where T
    println(io, "SphericalHarmonicBC{$T}")
    if isempty(bc.coeffs)
        _tree_row(io, "modes", "none"; last=true)
        return
    end
    pairs = sort(collect(bc.coeffs), by=x->(x[1][1], x[1][2]))
    for (i, ((ℓ, m), amp)) in enumerate(pairs)
        _tree_row(io, "Y_$ℓ,$m", amp; last=i == length(pairs))
    end
end


"""
    conduction_basic_state(cd::ChebyshevDiffn{T}, χ::T, lmax_bs::Int;
                           thermal_bc::Symbol=:fixed_temperature,
                           outer_flux::T=zero(T)) where T

Create a basic state corresponding to pure conduction (no meridional variation).

This is the default basic state with:
- θ̄(r) = conduction profile (only ℓ=0 component)
- ū_φ = 0 (no flow)

Arguments:
- `cd` - Chebyshev differentiation structure
- `χ` - Radius ratio r_i/r_o
- `lmax_bs` - Maximum ℓ for basic state (typically small, e.g., 4)
- `thermal_bc` - Outer thermal boundary condition:
  - `:fixed_temperature` (default): θ̄(r_o) = 0
  - `:fixed_flux`: dθ̄/dr|_{r_o} = outer_flux
- `outer_flux` - Heat flux at outer boundary (only used if thermal_bc=:fixed_flux)
                 Positive = heat flowing outward, negative = heat flowing inward.
                 For standard convection with hot inner boundary, use negative flux
                 (e.g., outer_flux = -1/(r_o² × (1 - χ)) for unit total flux).

# Physical Interpretation
For fixed temperature BCs:
  - θ̄(r_i) = 1 (hot inner boundary)
  - θ̄(r_o) = 0 (cold outer boundary)

For fixed flux at outer:
  - θ̄(r_i) = 1 (hot inner boundary)
  - dθ̄/dr|_{r_o} = outer_flux (prescribed heat flux)

The conduction profile for ℓ=0 with fixed flux at outer is:
  θ̄_0(r) = √(4π) × [1 + outer_flux × r_o² × (1/r_i - 1/r)]
"""
function conduction_basic_state(cd::ChebyshevDiffn{T}, χ::T, lmax_bs::Int;
                                thermal_bc::Symbol=:fixed_temperature,
                                outer_flux::T=zero(T)) where T
    r = cd.x
    Nr = length(r)

    r_i = T(χ)
    r_o = T(1.0)

    # Validate thermal BC
    if !(thermal_bc in (:fixed_temperature, :fixed_flux))
        error("thermal_bc must be :fixed_temperature or :fixed_flux, got: $thermal_bc")
    end

    # ℓ=0 conduction profile
    inner_value = sqrt(T(4) * T(pi))   # θ̄_00(r_i) = 1 × √(4π)

    if thermal_bc == :fixed_temperature
        outer_value = zero(T)           # θ̄_00(r_o) = 0
        theta_cond, dtheta_dr_cond = laplace_mode_profile(0, r, r_i, r_o,
                                                         inner_value, outer_value;
                                                         outer_bc=:fixed_temperature)
    else  # fixed_flux
        # For ℓ=0: dθ̄_00/dr|_{r_o} = outer_flux × √(4π)
        # (normalize by √(4π) to match the spherical harmonic coefficient)
        outer_flux_normalized = outer_flux * sqrt(T(4) * T(pi))
        theta_cond, dtheta_dr_cond = laplace_mode_profile(0, r, r_i, r_o,
                                                         inner_value, outer_flux_normalized;
                                                         outer_bc=:fixed_flux)
    end

    # Initialize dictionaries
    theta_coeffs = Dict{Int,Vector{T}}()
    uphi_coeffs = Dict{Int,Vector{T}}()
    dtheta_dr_coeffs = Dict{Int,Vector{T}}()
    duphi_dr_coeffs = Dict{Int,Vector{T}}()

    # Only ℓ=0 component is non-zero
    # Y_00 = 1/√(4π), so θ̄_00(r) = √(4π) × θ_cond(r)
    theta_coeffs[0] = theta_cond
    dtheta_dr_coeffs[0] = dtheta_dr_cond
    uphi_coeffs[0] = zeros(T, Nr)
    duphi_dr_coeffs[0] = zeros(T, Nr)

    # Higher ℓ modes are zero
    for ℓ in 1:lmax_bs
        theta_coeffs[ℓ] = zeros(T, Nr)
        dtheta_dr_coeffs[ℓ] = zeros(T, Nr)
        uphi_coeffs[ℓ] = zeros(T, Nr)
        duphi_dr_coeffs[ℓ] = zeros(T, Nr)
    end

    return BasicState(
        lmax_bs = lmax_bs,
        Nr = Nr,
        r = r,
        theta_coeffs = theta_coeffs,
        uphi_coeffs = uphi_coeffs,
        dtheta_dr_coeffs = dtheta_dr_coeffs,
        duphi_dr_coeffs = duphi_dr_coeffs
    )

end


"""
    meridional_basic_state(cd::ChebyshevDiffn{T}, χ::T, E::T, Ra::T, Pr::T,
                          lmax_bs::Int, amplitude::T;
                          mechanical_bc::Symbol=:no_slip,
                          thermal_bc::Symbol=:fixed_temperature,
                          outer_flux_mean::T=zero(T),
                          outer_flux_Y20::T=zero(T)) where T

Create a basic state with meridional temperature variation at the outer boundary.

# Fixed Temperature Boundary Conditions (default)
The inner boundary is held at uniform temperature:
    θ̄(r_i, θ) = 1

The outer boundary has zero-mean meridional variation:
    θ̄(r_o, θ) = amplitude × Y_20(θ)

This represents differential heating (e.g., equator hotter than poles).

# Fixed Flux Boundary Conditions
The inner boundary is held at uniform temperature:
    θ̄(r_i, θ) = 1

The outer boundary has prescribed heat flux:
    dθ̄/dr|_{r_o} = outer_flux_mean + outer_flux_Y20 × Y_20(θ)

This represents:
- `outer_flux_mean` : Mean (spherically symmetric) heat flux at outer boundary
- `outer_flux_Y20` : Meridional variation in heat flux (amplitude of Y_20 component)

# Arguments
- `cd` - Chebyshev differentiation structure
- `χ` - Radius ratio r_i/r_o
- `E` - Ekman number (REQUIRED for thermal wind balance scaling)
- `Ra` - Rayleigh number (needed for thermal wind balance)
- `Pr` - Prandtl number
- `lmax_bs` - Maximum ℓ for basic state expansion
- `amplitude` - For :fixed_temperature: amplitude of Y_20 temperature at outer boundary
               For :fixed_flux: amplitude of Y_20 FLUX at outer boundary (overrides outer_flux_Y20)
- `mechanical_bc` - Mechanical BCs: `:no_slip` (default) or `:stress_free`
- `thermal_bc` - Outer thermal BC: `:fixed_temperature` (default) or `:fixed_flux`
- `outer_flux_mean` - Mean heat flux at outer (for :fixed_flux only)
- `outer_flux_Y20` - Y_20 flux amplitude at outer (for :fixed_flux only, overridden by amplitude)

# Physical Examples

1. Differential heating with fixed temperatures:
   ```julia
   bs = meridional_basic_state(cd, χ, E, Ra, Pr, 4, 0.1)
   # θ̄(r_i) = 1 (uniform hot), θ̄(r_o) = 0.1 × Y_20 (equator warmer than poles)
   ```

2. Fixed inner temperature, uniform flux out:
   ```julia
   bs = meridional_basic_state(cd, χ, E, Ra, Pr, 4, 0.0;
                               thermal_bc=:fixed_flux,
                               outer_flux_mean=-1.0)
   # θ̄(r_i) = 1, dθ̄/dr|_{r_o} = -1 (heat flowing out uniformly)
   ```

3. Fixed inner temperature, meridionally-varying flux:
   ```julia
   bs = meridional_basic_state(cd, χ, E, Ra, Pr, 4, 0.1;
                               thermal_bc=:fixed_flux,
                               outer_flux_mean=-1.0)
   # θ̄(r_i) = 1, dθ̄/dr|_{r_o} = -1 + 0.1 × Y_20 (more heat loss at equator)
   ```
"""
function meridional_basic_state(cd::ChebyshevDiffn{T}, χ::T, E::T, Ra::T, Pr::T,
                               lmax_bs::Int, amplitude::T;
                               mechanical_bc::Symbol=:no_slip,
                               thermal_bc::Symbol=:fixed_temperature,
                               outer_flux_mean::T=zero(T),
                               outer_flux_Y20::T=zero(T)) where T

    r = cd.x
    Nr = length(r)
    r_i = T(χ)
    r_o = T(1.0)

    # Validate BCs
    if !(thermal_bc in (:fixed_temperature, :fixed_flux))
        error("thermal_bc must be :fixed_temperature or :fixed_flux, got: $thermal_bc")
    end

    # Initialize dictionaries
    theta_coeffs = Dict{Int,Vector{T}}()
    dtheta_dr_coeffs = Dict{Int,Vector{T}}()
    uphi_coeffs = Dict{Int,Vector{T}}()
    duphi_dr_coeffs = Dict{Int,Vector{T}}()

    # Spherical harmonic normalization for Y_20
    norm_Y20 = sqrt(T(5) / (T(4) * T(pi)))

    if thermal_bc == :fixed_temperature
        # =====================================================================
        # Fixed temperature at both boundaries
        # =====================================================================

        # ℓ=0 mode: uniform inner temp, zero outer temp
        theta_0, dtheta_0 = laplace_mode_profile(0, r, r_i, r_o,
                                                 sqrt(T(4)*T(pi)), zero(T);
                                                 outer_bc=:fixed_temperature)
        theta_coeffs[0] = theta_0
        dtheta_dr_coeffs[0] = dtheta_0

        # ℓ=2 mode: zero at inner, amplitude × Y_20 pattern at outer
        theta_2, dtheta_2 = laplace_mode_profile(2, r, r_i, r_o,
                                                zero(T), amplitude / norm_Y20;
                                                outer_bc=:fixed_temperature)
        theta_coeffs[2] = theta_2
        dtheta_dr_coeffs[2] = dtheta_2

    else  # fixed_flux
        # =====================================================================
        # Fixed temperature at inner, fixed flux at outer
        # =====================================================================

        # ℓ=0 mode: uniform inner temp, prescribed mean flux at outer
        # Normalize flux by √(4π) for spherical harmonic coefficient
        flux_0_normalized = outer_flux_mean * sqrt(T(4) * T(pi))
        theta_0, dtheta_0 = laplace_mode_profile(0, r, r_i, r_o,
                                                 sqrt(T(4)*T(pi)), flux_0_normalized;
                                                 outer_bc=:fixed_flux)
        theta_coeffs[0] = theta_0
        dtheta_dr_coeffs[0] = dtheta_0

        # ℓ=2 mode: zero at inner, prescribed Y_20 flux at outer
        # Use amplitude as the flux amplitude (overrides outer_flux_Y20 if non-zero)
        flux_Y20 = amplitude != zero(T) ? amplitude : outer_flux_Y20
        flux_2_normalized = flux_Y20 / norm_Y20
        theta_2, dtheta_2 = laplace_mode_profile(2, r, r_i, r_o,
                                                zero(T), flux_2_normalized;
                                                outer_bc=:fixed_flux)
        theta_coeffs[2] = theta_2
        dtheta_dr_coeffs[2] = dtheta_2
    end

    # Initialize velocity modes to zero
    uphi_coeffs[0] = zeros(T, Nr)
    duphi_dr_coeffs[0] = zeros(T, Nr)
    uphi_coeffs[2] = zeros(T, Nr)
    duphi_dr_coeffs[2] = zeros(T, Nr)

    # =========================================================================
    # Higher ℓ modes: zero (no higher-order boundary variations)
    # =========================================================================
    for ℓ in 1:lmax_bs
        if ℓ == 0 || ℓ == 2
            continue
        end
        theta_coeffs[ℓ] = zeros(T, Nr)
        dtheta_dr_coeffs[ℓ] = zeros(T, Nr)
        uphi_coeffs[ℓ] = zeros(T, Nr)
        duphi_dr_coeffs[ℓ] = zeros(T, Nr)
    end

    # =========================================================================
    # Solve thermal wind balance for ū_φ
    # =========================================================================
    # Use the full coupled operator (not the diagonal heuristic): the diagonal
    # solver placed the zonal flow at the wrong (odd-L) parity and did not satisfy
    # the thermal-wind PDE. The coupled solve (m_bs=0) yields the correct
    # equatorially-symmetric (even-L) flow that satisfies the balance.
    solve_thermal_wind_coupled!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs, 0,
                                cd, r_i, r_o, Ra, Pr;
                                mechanical_bc=mechanical_bc,
                                E=E)

    return BasicState(
        lmax_bs = lmax_bs,
        Nr = Nr,
        r = r,
        theta_coeffs = theta_coeffs,
        uphi_coeffs = uphi_coeffs,
        dtheta_dr_coeffs = dtheta_dr_coeffs,
        duphi_dr_coeffs = duphi_dr_coeffs
    )
end


# Helper: coefficients for expanding derivatives of Legendre polynomials.
#
# Returns a vector `deriv_maps` where `deriv_maps[ℓ]` is a dictionary mapping
# target degree L to the coefficient c_{ℓ,L} in
#     P_ℓ'(x) = Σ c_{ℓ,L} P_L(x)
# with L ranging over ℓ-1, ℓ-3, … (same parity as ℓ-1).
"""Compute Legendre derivative expansion maps up to degree `lmax`."""
function legendre_derivative_coefficients(lmax::Int)
    maps = Dict{Int, Dict{Int,Float64}}()
    maps[0] = Dict{Int,Float64}()           # P₀' = 0
    if lmax >= 1
        maps[1] = Dict(0 => 1.0)            # P₁' = P₀
    end

    for ℓ in 2:lmax
        coeffs = Dict{Int,Float64}()
        coeffs[ℓ-1] = (2ℓ - 1) * 1.0        # (2ℓ-1) P_{ℓ-1}

        for (k, v) in maps[ℓ-2]
            coeffs[k] = get(coeffs, k, 0.0) + v
        end

        maps[ℓ] = coeffs
    end

    return maps
end


"""
    laplace_mode_profile(ℓ, r, r_i, r_o, inner_value, outer_value;
                         outer_bc=:fixed_temperature)

Solve the radial Laplace equation for spherical harmonic mode ℓ.

The equation is: d²θ̄_ℓ/dr² + (2/r) dθ̄_ℓ/dr - ℓ(ℓ+1)/r² θ̄_ℓ = 0

General solution: θ̄_ℓ(r) = A r^ℓ + B r^{-(ℓ+1)}

# Arguments
- `ℓ::Int` - Spherical harmonic degree
- `r` - Radial grid points
- `r_i, r_o` - Inner and outer radii
- `inner_value` - Value or flux at inner boundary (always Dirichlet for temperature)
- `outer_value` - Value (for :fixed_temperature) or flux (for :fixed_flux) at outer boundary
- `outer_bc` - Outer boundary condition type:
  - `:fixed_temperature` (default): θ̄_ℓ(r_o) = outer_value
  - `:fixed_flux`: dθ̄_ℓ/dr|_{r_o} = outer_value

# Returns
- `θ` - Temperature profile θ̄_ℓ(r)
- `dθ` - Radial derivative dθ̄_ℓ/dr

# Mathematical Details
For fixed temperature at both boundaries:
  - θ̄_ℓ(r_i) = inner_value
  - θ̄_ℓ(r_o) = outer_value

For fixed temperature at inner, fixed flux at outer:
  - θ̄_ℓ(r_i) = inner_value
  - dθ̄_ℓ/dr|_{r_o} = outer_value

The derivative is: dθ̄_ℓ/dr = A ℓ r^{ℓ-1} - B (ℓ+1) r^{-(ℓ+2)}
"""
function laplace_mode_profile(ℓ::Int, r::AbstractVector{T}, r_i::T, r_o::T,
                             inner_value::T, outer_value::T;
                             outer_bc::Symbol=:fixed_temperature) where T

    if outer_bc == :fixed_temperature
        # Both boundaries have Dirichlet conditions (fixed temperature)
        # θ̄_ℓ(r_i) = inner_value
        # θ̄_ℓ(r_o) = outer_value
        M = T[
            r_i^ℓ          r_i^(-(ℓ+1));
            r_o^ℓ          r_o^(-(ℓ+1))
        ]
        rhs = T[inner_value, outer_value]

    elseif outer_bc == :fixed_flux
        # Inner: Dirichlet (fixed temperature)
        # Outer: Neumann (fixed flux)
        # θ̄_ℓ(r_i) = inner_value
        # dθ̄_ℓ/dr|_{r_o} = outer_value
        #
        # From θ̄_ℓ = A r^ℓ + B r^{-(ℓ+1)}:
        #   dθ̄_ℓ/dr = A ℓ r^{ℓ-1} - B (ℓ+1) r^{-(ℓ+2)}
        #
        # At r = r_o:
        #   dθ̄_ℓ/dr|_{r_o} = A ℓ r_o^{ℓ-1} - B (ℓ+1) r_o^{-(ℓ+2)}
        M = T[
            r_i^ℓ                    r_i^(-(ℓ+1));
            ℓ * r_o^(ℓ-1)           -(ℓ+1) * r_o^(-(ℓ+2))
        ]
        rhs = T[inner_value, outer_value]

    else
        error("outer_bc must be :fixed_temperature or :fixed_flux, got: $outer_bc")
    end

    α, β = M \ rhs

    θ = α .* r.^ℓ .+ β .* r.^(-(ℓ+1))
    dθ = α * ℓ .* r.^(ℓ-1) .- β * (ℓ+1) .* r.^(-(ℓ+2))

    return θ, dθ
end


# Note: The solve_thermal_wind_balance! function with E parameter is defined below
# (after the non-axisymmetric basic state functions)


"""
    evaluate_basic_state(bs::BasicState{T}, r_eval::T, theta_eval::T) where T

Evaluate the basic state at a given (r, θ) point.

Returns:
- `theta_bar` - Temperature θ̄(r,θ)
- `uphi_bar` - Zonal velocity ū_φ(r,θ)
- `dtheta_dr` - Radial derivative ∂θ̄/∂r
- `dtheta_dtheta` - Meridional derivative ∂θ̄/∂θ
- `duphi_dr` - Radial derivative ∂ū_φ/∂r
- `duphi_dtheta` - Meridional derivative ∂ū_φ/∂θ
"""
function evaluate_basic_state(bs::BasicState{T}, r_eval::T, theta_eval::T) where T
    rmin = min(first(bs.r), last(bs.r))
    rmax = max(first(bs.r), last(bs.r))
    if r_eval < rmin || r_eval > rmax
        throw(ArgumentError("r_eval must be within [$rmin, $rmax]"))
    end

    lmax = bs.lmax_bs
    x = cos(theta_eval)
    P, dPdx = _legendre_values_and_derivs(lmax, x)
    sinθ = sin(theta_eval)

    norms = Vector{T}(undef, lmax + 1)
    for ℓ in 0:lmax
        norms[ℓ + 1] = sqrt(T(2 * ℓ + 1) / (T(4) * T(pi)))
    end

    theta_bar = zero(T)
    uphi_bar = zero(T)
    dtheta_dr = zero(T)
    dtheta_dtheta = zero(T)
    duphi_dr = zero(T)
    duphi_dtheta = zero(T)

    for (ℓ, coeffs) in bs.theta_coeffs
        ℓ > lmax && continue
        coeff = _linear_interpolate(bs.r, coeffs, r_eval)
        Y = norms[ℓ + 1] * P[ℓ + 1]
        dY_dtheta = -sinθ * norms[ℓ + 1] * dPdx[ℓ + 1]
        theta_bar += coeff * Y
        dtheta_dtheta += coeff * dY_dtheta
        if haskey(bs.dtheta_dr_coeffs, ℓ)
            dtheta_dr += _linear_interpolate(bs.r, bs.dtheta_dr_coeffs[ℓ], r_eval) * Y
        end
    end

    for (ℓ, coeffs) in bs.uphi_coeffs
        ℓ > lmax && continue
        coeff = _linear_interpolate(bs.r, coeffs, r_eval)
        Y = norms[ℓ + 1] * P[ℓ + 1]
        dY_dtheta = -sinθ * norms[ℓ + 1] * dPdx[ℓ + 1]
        uphi_bar += coeff * Y
        duphi_dtheta += coeff * dY_dtheta
        if haskey(bs.duphi_dr_coeffs, ℓ)
            duphi_dr += _linear_interpolate(bs.r, bs.duphi_dr_coeffs[ℓ], r_eval) * Y
        end
    end

    return (
        theta_bar = theta_bar,
        uphi_bar = uphi_bar,
        dtheta_dr = dtheta_dr,
        dtheta_dtheta = dtheta_dtheta,
        duphi_dr = duphi_dr,
        duphi_dtheta = duphi_dtheta
    )
end

"""Return Legendre values and x-derivatives up to `lmax` at one point."""
function _legendre_values_and_derivs(lmax::Int, x::T) where T
    P = zeros(T, lmax + 1)
    dPdx = zeros(T, lmax + 1)
    P[1] = one(T)
    if lmax >= 1
        P[2] = x
    end
    for l in 2:lmax
        P[l + 1] = ((2 * l - 1) * x * P[l] - (l - 1) * P[l - 1]) / l
    end

    dPdx[1] = zero(T)
    if lmax >= 1
        denom = one(T) - x * x
        tol = sqrt(eps(T))
        for l in 1:lmax
            if abs(denom) < tol
                dPdx[l + 1] = zero(T)
            else
                dPdx[l + 1] = l * (P[l] - x * P[l + 1]) / denom
            end
        end
    end

    return P, dPdx
end

"""Linearly interpolate radial data regardless of whether the grid ascends or descends."""
function _linear_interpolate(r::AbstractVector{T}, values::AbstractVector{T}, r_eval::T) where T
    length(r) == length(values) || throw(DimensionMismatch("r and values must have same length"))
    if r[1] <= r[end]
        return _linear_interpolate_ascending(r, values, r_eval)
    end
    r_rev = reverse(r)
    values_rev = reverse(values)
    return _linear_interpolate_ascending(r_rev, values_rev, r_eval)
end

"""Linearly interpolate on an ascending radial grid with endpoint clamping."""
function _linear_interpolate_ascending(r::AbstractVector{T}, values::AbstractVector{T}, r_eval::T) where T
    n = length(r)
    r_eval <= r[1] && return values[1]
    r_eval >= r[end] && return values[end]
    j = searchsortedlast(r, r_eval)
    j == n && (j = n - 1)
    t = (r_eval - r[j]) / (r[j + 1] - r[j])
    return (one(T) - t) * values[j] + t * values[j + 1]
end


# =============================================================================
#  Non-Axisymmetric (3D) Basic States
# =============================================================================

"""
    nonaxisymmetric_basic_state(cd::ChebyshevDiffn, χ::Real, E::Real, Ra::Real, Pr::Real,
                                lmax_bs::Int, mmax_bs::Int,
                                amplitudes::AbstractDict;
                                mechanical_bc::Symbol=:no_slip,
                                thermal_bc::Symbol=:fixed_temperature,
                                outer_fluxes::AbstractDict=Dict{Tuple{Int,Int},Float64}())

Create a 3D basic state with both meridional and longitudinal temperature variations.

# Fixed Temperature Boundary Conditions (default)
The inner boundary is held at uniform temperature:
    θ̄(r_i, θ, φ) = 1

The outer boundary has zero-mean variations:
    θ̄(r_o, θ, φ) = Σ_{ℓ,m} amplitude_{ℓm} × Y_ℓm(θ,φ)

# Fixed Flux Boundary Conditions
The inner boundary is held at uniform temperature:
    θ̄(r_i, θ, φ) = 1

The outer boundary has prescribed heat flux:
    dθ̄/dr|_{r_o} = Σ_{ℓ,m} flux_{ℓm} × Y_ℓm(θ,φ)

where `flux_{ℓm}` values are taken from `outer_fluxes` dictionary, and `amplitudes`
provides the flux values for any (ℓ,m) not in `outer_fluxes`.

This represents fully 3D differential heating scenarios, such as:
- Longitudinally-varying solar heating
- Zonal wavenumber patterns in thermal forcing
- Realistic planetary/stellar boundary conditions

The interior temperature θ̄(r,θ,φ) satisfies ∇²θ̄ = 0 with these BCs.

The zonal velocity field ū_φ(r,θ,φ) is computed from the coupled thermal wind
balance by default. Pass `coupled_thermal_wind=false` to use the faster diagonal
approximation.

# Arguments
- `cd` - Chebyshev differentiation structure
- `χ` - Radius ratio r_i/r_o
- `E` - Ekman number (REQUIRED for thermal wind balance scaling)
- `Ra` - Rayleigh number (for thermal wind balance)
- `Pr` - Prandtl number
- `lmax_bs` - Maximum ℓ for basic state
- `mmax_bs` - Maximum m for basic state (e.g., 0-4)
- `amplitudes` - Dict{(ℓ,m) => value} specifying:
  - For :fixed_temperature: boundary temperature amplitudes
  - For :fixed_flux: flux amplitudes (if not specified in outer_fluxes)
- `mechanical_bc` - Mechanical boundary conditions: `:no_slip` (default) or `:stress_free`
- `thermal_bc` - Outer thermal BC: `:fixed_temperature` (default) or `:fixed_flux`
- `outer_fluxes` - Dict{(ℓ,m) => flux} specifying heat flux at outer boundary
                   (only used if thermal_bc=:fixed_flux)
- `coupled_thermal_wind` - Use full coupled thermal-wind solve (default: true)
- `include_meridional_flow` - Compute ū_r and ū_θ for m≠0 from geostrophic
                              balance (default: true)

# Examples

1. Fixed temperature at outer boundary (standard differential heating):
```julia
amplitudes = Dict(
    (2,0) => 0.1,   # Meridional Y_20 temperature pattern
    (2,2) => 0.05   # Longitudinal Y_22 temperature pattern
)
bs3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 2, amplitudes)
```

2. Fixed flux at outer boundary:
```julia
outer_fluxes = Dict(
    (0,0) => -1.0,   # Mean heat flux (negative = outward)
    (2,0) => 0.1,    # Meridional flux variation
    (2,2) => 0.05    # Longitudinal flux variation
)
bs3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 2, Dict{Tuple{Int,Int},Float64}();
                                    thermal_bc=:fixed_flux, outer_fluxes=outer_fluxes)
```

3. Mixed specification (amplitudes provide defaults for flux):
```julia
amplitudes = Dict((2,0) => 0.1, (2,2) => 0.05)  # Treated as flux amplitudes
bs3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 2, amplitudes;
                                    thermal_bc=:fixed_flux)
```
"""
function nonaxisymmetric_basic_state(cd::ChebyshevDiffn, χ::Real, E::Real, Ra::Real, Pr::Real,
                                     lmax_bs::Int, mmax_bs::Int,
                                     amplitudes::AbstractDict;
                                     mechanical_bc::Symbol=:no_slip,
                                     thermal_bc::Symbol=:fixed_temperature,
                                     outer_fluxes::AbstractDict=Dict{Tuple{Int,Int},Float64}(),
                                     coupled_thermal_wind::Bool=true,
                                     include_meridional_flow::Bool=true)

    r = cd.x
    T = eltype(r)  # Get the element type from the Chebyshev grid
    Nr = length(r)
    r_i = T(χ)
    r_o = T(1.0)

    # Validate thermal BC
    if !(thermal_bc in (:fixed_temperature, :fixed_flux))
        error("thermal_bc must be :fixed_temperature or :fixed_flux, got: $thermal_bc")
    end

    # Initialize all coefficient dictionaries
    theta_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    dtheta_dr_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    ur_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    utheta_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    uphi_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    dur_dr_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    dutheta_dr_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    duphi_dr_coeffs = Dict{Tuple{Int,Int},Vector{T}}()

    # Spherical harmonic normalization
    # For m=0: norm = sqrt((2ℓ+1)/(4π))
    # For m≠0: norm = sqrt((2ℓ+1)/(4π) × 2)
    Y_norm(ℓ::Int, m::Int) = m == 0 ? sqrt(T(2ℓ+1)/(4*T(π))) : sqrt(T(2ℓ+1)/(4*T(π)) * 2)

    # =========================================================================
    # Solve ∇²θ̄ = 0 for each (ℓ,m) mode
    # =========================================================================

    for ℓ in 0:lmax_bs
        for m in 0:min(ℓ, mmax_bs)
            norm_Ylm = Y_norm(ℓ, m)

            if ℓ == 0 && m == 0
                # =============================================================
                # ℓ=0, m=0: Radial conduction profile (mean temperature)
                # =============================================================
                # Inner BC: θ̄_00(r_i) = 1 × √(4π) (uniform temperature = 1)
                inner_value = sqrt(T(4) * T(π))

                if thermal_bc == :fixed_temperature
                    # Outer BC: θ̄_00(r_o) = 0 (cold outer boundary)
                    outer_value = zero(T)
                    theta_00, dtheta_00 = laplace_mode_profile(0, r, r_i, r_o,
                                                               inner_value, outer_value;
                                                               outer_bc=:fixed_temperature)
                else  # fixed_flux
                    # Outer BC: dθ̄_00/dr|_{r_o} = flux_00 × √(4π)
                    # Get flux from outer_fluxes or amplitudes
                    flux_00 = get(outer_fluxes, (0,0), get(amplitudes, (0,0), zero(T)))
                    outer_flux_normalized = T(flux_00) * sqrt(T(4) * T(π))
                    theta_00, dtheta_00 = laplace_mode_profile(0, r, r_i, r_o,
                                                               inner_value, outer_flux_normalized;
                                                               outer_bc=:fixed_flux)
                end

                theta_coeffs[(0,0)] = theta_00
                dtheta_dr_coeffs[(0,0)] = dtheta_00

            else
                # =============================================================
                # ℓ > 0 or m > 0: Higher-order modes
                # =============================================================
                # Get amplitude/flux for this mode
                # For fixed_flux: check outer_fluxes first, then amplitudes
                if thermal_bc == :fixed_flux
                    value = get(outer_fluxes, (ℓ,m), get(amplitudes, (ℓ,m), zero(T)))
                else
                    value = get(amplitudes, (ℓ,m), zero(T))
                end

                if value != 0
                    # Inner BC: θ̄_ℓm(r_i) = 0 (uniform inner temperature)
                    inner_value = zero(T)

                    if thermal_bc == :fixed_temperature
                        # Outer BC: θ̄_ℓm(r_o) = amplitude / norm_Ylm
                        outer_value = T(value) / norm_Ylm
                        theta_lm, dtheta_lm = laplace_mode_profile(ℓ, r, r_i, r_o,
                                                                   inner_value, outer_value;
                                                                   outer_bc=:fixed_temperature)
                    else  # fixed_flux
                        # Outer BC: dθ̄_ℓm/dr|_{r_o} = flux / norm_Ylm
                        outer_flux_normalized = T(value) / norm_Ylm
                        theta_lm, dtheta_lm = laplace_mode_profile(ℓ, r, r_i, r_o,
                                                                   inner_value, outer_flux_normalized;
                                                                   outer_bc=:fixed_flux)
                    end

                    theta_coeffs[(ℓ,m)] = theta_lm
                    dtheta_dr_coeffs[(ℓ,m)] = dtheta_lm

                else
                    # Zero value: Initialize to zero
                    theta_coeffs[(ℓ,m)] = zeros(T, Nr)
                    dtheta_dr_coeffs[(ℓ,m)] = zeros(T, Nr)
                end
            end

            # Initialize velocity components to zero (will be filled by thermal wind)
            ur_coeffs[(ℓ,m)] = zeros(T, Nr)
            utheta_coeffs[(ℓ,m)] = zeros(T, Nr)
            uphi_coeffs[(ℓ,m)] = zeros(T, Nr)
            dur_dr_coeffs[(ℓ,m)] = zeros(T, Nr)
            dutheta_dr_coeffs[(ℓ,m)] = zeros(T, Nr)
            duphi_dr_coeffs[(ℓ,m)] = zeros(T, Nr)
        end
    end

    # =========================================================================
    # Solve thermal wind balance for ALL azimuthal modes
    # =========================================================================
    # For non-axisymmetric temperature variations Y_ℓm with m≠0, the thermal
    # wind balance generates velocity components that also have azimuthal
    # structure. The full geostrophic balance is:
    #
    #   2Ω × ū = -∇p + Ra E² Θ̄ g r̂ / Pr
    #
    # For the φ-component (thermal wind):
    #   cos(θ) ∂ū_φ/∂r - sin(θ) ū_φ/r = -(Ra E²)/(2Pr) × (r/r_o) × (1/r) × ∂Θ̄/∂θ
    #
    # For m≠0 modes, we also get contributions from ∂Θ̄/∂φ to ū_θ, but the
    # leading order balance for ū_φ follows the same thermal wind structure.
    #
    # Key insight: ∂Y_ℓm/∂θ couples to Y_{ℓ±1,m} (same m, different ℓ)
    # So Y_22 temperature generates velocity in Y_12 and Y_32 modes.

    # Process each azimuthal wavenumber m separately
    for m_bs in 0:mmax_bs
        # Extract temperature modes for this m
        theta_m = Dict{Int, Vector{T}}()
        for ℓ in m_bs:lmax_bs  # ℓ ≥ m required
            if haskey(theta_coeffs, (ℓ, m_bs))
                theta_m[ℓ] = theta_coeffs[(ℓ, m_bs)]
            end
        end

        # Skip if no temperature modes for this m
        if isempty(theta_m) || all(maximum(abs, v) < 1e-15 for v in values(theta_m))
            continue
        end

        # Initialize velocity storage for this m
        uphi_m = Dict{Int, Vector{T}}(ℓ => zeros(T, Nr) for ℓ in 0:lmax_bs)
        duphi_dr_m = Dict{Int, Vector{T}}(ℓ => zeros(T, Nr) for ℓ in 0:lmax_bs)

        # Solve thermal wind for this azimuthal mode
        if coupled_thermal_wind
            # Full coupled solver (no diagonal approximation)
            solve_thermal_wind_coupled!(uphi_m, duphi_dr_m, theta_m, m_bs,
                                        cd, r_i, r_o, Ra, Pr;
                                        mechanical_bc=mechanical_bc,
                                        E=E, lmax=lmax_bs + 1)
        else
            # Diagonal approximation (faster but less accurate)
            solve_thermal_wind_balance_3d!(uphi_m, duphi_dr_m, theta_m, m_bs,
                                           cd, r_i, r_o, Ra, Pr;
                                           mechanical_bc=mechanical_bc,
                                           E=E)
        end

        # Copy results to 3D storage
        for ℓ in 0:lmax_bs
            if haskey(uphi_m, ℓ) && maximum(abs, uphi_m[ℓ]) > 1e-15
                uphi_coeffs[(ℓ, m_bs)] = uphi_m[ℓ]
                duphi_dr_coeffs[(ℓ, m_bs)] = duphi_dr_m[ℓ]
            end
        end
    end

    if include_meridional_flow
        solve_meridional_circulation_toroidal_poloidal!(
            ur_coeffs, utheta_coeffs, dur_dr_coeffs, dutheta_dr_coeffs,
            theta_coeffs, uphi_coeffs,
            r, Matrix(cd.D1), Matrix(cd.D2), r_i, r_o,
            T(Ra), T(E), T(Pr), lmax_bs, mmax_bs;
            mechanical_bc=mechanical_bc,
            include_meridional=true,
            use_full_coupling=true,
        )
    end

    return BasicState3D(
        lmax_bs = lmax_bs,
        mmax_bs = mmax_bs,
        Nr = Nr,
        r = r,
        theta_coeffs = theta_coeffs,
        dtheta_dr_coeffs = dtheta_dr_coeffs,
        ur_coeffs = ur_coeffs,
        utheta_coeffs = utheta_coeffs,
        uphi_coeffs = uphi_coeffs,
        dur_dr_coeffs = dur_dr_coeffs,
        dutheta_dr_coeffs = dutheta_dr_coeffs,
        duphi_dr_coeffs = duphi_dr_coeffs
    )
end


# =============================================================================
#  Convenience Function: basic_state with Symbolic BCs
#
#  High-level interface that accepts SphericalHarmonicBC objects and
#  automatically dispatches to the appropriate low-level function.
# =============================================================================

"""
    basic_state(cd, χ, E, Ra, Pr;
                temperature_bc=nothing,
                flux_bc=nothing,
                mechanical_bc=:no_slip,
                lmax_bs=nothing)

Create a basic state with symbolic spherical harmonic boundary conditions.

This is a convenience wrapper that accepts `SphericalHarmonicBC` objects
(created with `Y20()`, `Y22()`, etc.) and automatically dispatches to the
appropriate low-level function (`conduction_basic_state`, `meridional_basic_state`,
or `nonaxisymmetric_basic_state`).

# Arguments
- `cd` : ChebyshevDiffn - Chebyshev differentiation structure
- `χ` : Radius ratio r_i/r_o
- `E` : Ekman number
- `Ra` : Rayleigh number
- `Pr` : Prandtl number

# Keyword Arguments
- `temperature_bc` : SphericalHarmonicBC specifying temperature at outer boundary
- `flux_bc` : SphericalHarmonicBC specifying heat flux at outer boundary
  (Cannot specify both temperature_bc and flux_bc)
- `mechanical_bc` : `:no_slip` (default) or `:stress_free`
- `lmax_bs` : Maximum ℓ for basic state (auto-determined if not specified)

# Returns
- `BasicState` if boundary condition is axisymmetric (m=0 only)
- `BasicState3D` if boundary condition has m≠0 components

# Examples

## Pure conduction (no temperature variation at outer boundary)
```julia
bs = basic_state(cd, χ, E, Ra, Pr)
```

## Meridional temperature variation (Y₂₀ pattern)
```julia
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=Y20(0.1))
```

## Combined meridional and longitudinal variation
```julia
bc = Y20(0.1) + Y22(0.05)
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)
```

## Fixed flux at outer boundary
```julia
# Uniform outward heat flux plus meridional variation
flux = Y00(-1.0) + Y20(0.1)
bs = basic_state(cd, χ, E, Ra, Pr; flux_bc=flux)
```

## Stress-free boundaries with temperature variation
```julia
bs = basic_state(cd, χ, E, Ra, Pr;
                 temperature_bc=Y20(0.1),
                 mechanical_bc=:stress_free)
```

# Automatic Dispatch Logic

The function automatically selects the appropriate implementation:

1. If `temperature_bc` is `nothing` (or only Y00 component) and no `flux_bc`:
   → `conduction_basic_state` (pure conduction profile)

2. If boundary condition is axisymmetric (only m=0 modes):
   → `meridional_basic_state` (returns `BasicState`)

3. If boundary condition has m≠0 modes:
   → `nonaxisymmetric_basic_state` (returns `BasicState3D`)
"""
function basic_state(cd, χ::Real, E::Real, Ra::Real, Pr::Real;
                     temperature_bc::Union{Nothing, SphericalHarmonicBC}=nothing,
                     flux_bc::Union{Nothing, SphericalHarmonicBC}=nothing,
                     mechanical_bc::Symbol=:no_slip,
                     lmax_bs::Union{Nothing, Int}=nothing,
                     coupled_thermal_wind::Bool=true)

    # Validate: can't have both temperature_bc and flux_bc
    if temperature_bc !== nothing && flux_bc !== nothing
        error("Cannot specify both temperature_bc and flux_bc. Choose one.")
    end

    T = eltype(cd.x)

    # Determine thermal BC type and the boundary condition
    if flux_bc !== nothing
        thermal_bc = :fixed_flux
        bc = flux_bc
    elseif temperature_bc !== nothing
        thermal_bc = :fixed_temperature
        bc = temperature_bc
    else
        # No BC specified → pure conduction
        _lmax = lmax_bs === nothing ? 4 : lmax_bs
        return conduction_basic_state(cd, T(χ), _lmax;
                                      thermal_bc=:fixed_temperature)
    end

    # Get lmax and mmax from boundary condition
    bc_lmax, bc_mmax = get_lmax_mmax(bc)

    # Use provided lmax_bs or auto-determine (add 2 for thermal wind coupling)
    _lmax = lmax_bs === nothing ? max(bc_lmax + 2, 4) : lmax_bs

    # Check if BC is effectively zero (only conduction)
    if iszero(bc)
        return conduction_basic_state(cd, T(χ), _lmax; thermal_bc=thermal_bc)
    end

    # Check if BC is purely Y00 (uniform outer temperature/flux)
    if bc_lmax == 0 && bc_mmax == 0
        # Only Y00 component - use conduction with outer_flux if flux BC
        if thermal_bc == :fixed_flux
            outer_flux_val = get(bc.coeffs, (0,0), zero(T))
            return conduction_basic_state(cd, T(χ), _lmax;
                                          thermal_bc=:fixed_flux,
                                          outer_flux=T(outer_flux_val))
        else
            # temperature_bc with only Y00 is unusual but valid
            # Use meridional with zero Y20 amplitude
            return meridional_basic_state(cd, T(χ), T(E), T(Ra), T(Pr),
                                          _lmax, zero(T);
                                          mechanical_bc=mechanical_bc,
                                          thermal_bc=:fixed_temperature)
        end
    end

    # Check if axisymmetric (m=0 only) → use meridional_basic_state
    if is_axisymmetric(bc)
        # Extract Y20 amplitude for meridional_basic_state
        amp_Y20 = get(bc.coeffs, (2,0), zero(T))

        if thermal_bc == :fixed_temperature
            return meridional_basic_state(cd, T(χ), T(E), T(Ra), T(Pr),
                                          _lmax, T(amp_Y20);
                                          mechanical_bc=mechanical_bc,
                                          thermal_bc=:fixed_temperature)
        else  # fixed_flux
            # Get mean flux (Y00) and Y20 flux
            flux_mean = get(bc.coeffs, (0,0), zero(T))
            flux_Y20 = get(bc.coeffs, (2,0), zero(T))
            return meridional_basic_state(cd, T(χ), T(E), T(Ra), T(Pr),
                                          _lmax, zero(T);  # amplitude=0 since we use explicit fluxes
                                          mechanical_bc=mechanical_bc,
                                          thermal_bc=:fixed_flux,
                                          outer_flux_mean=T(flux_mean),
                                          outer_flux_Y20=T(flux_Y20))
        end
    end

    # Non-axisymmetric → use nonaxisymmetric_basic_state
    amplitudes = to_dict(bc)

    if thermal_bc == :fixed_temperature
        return nonaxisymmetric_basic_state(cd, T(χ), T(E), T(Ra), T(Pr),
                                           _lmax, bc_mmax, amplitudes;
                                           mechanical_bc=mechanical_bc,
                                           thermal_bc=:fixed_temperature,
                                           coupled_thermal_wind=coupled_thermal_wind)
    else  # fixed_flux
        return nonaxisymmetric_basic_state(cd, T(χ), T(E), T(Ra), T(Pr),
                                           _lmax, bc_mmax,
                                           Dict{Tuple{Int,Int},T}();  # empty amplitudes
                                           mechanical_bc=mechanical_bc,
                                           thermal_bc=:fixed_flux,
                                           outer_fluxes=amplitudes,
                                           coupled_thermal_wind=coupled_thermal_wind)
    end
end


# =============================================================================
#  CORRECTED Thermal Wind Balance Solver
#
#  Key fixes:
#  1. Added missing E² factor in the prefactor
#  2. Corrected spherical harmonic coupling for ∂Y_ℓ0/∂θ
#  3. Fixed boundary condition application
#  4. Proper Chebyshev spectral BVP solver (replaces trapezoidal integration)
#  5. Full stress-free boundary condition support
# =============================================================================

function _thermal_wind_operator_lu(r::AbstractVector{T}, D1::AbstractMatrix{T},
                                   idx_inner::Int, mechanical_bc::Symbol) where {T<:Real}
    Nr = length(r)
    A_mat = Matrix{T}(undef, Nr, Nr)

    @inbounds for j in 1:Nr, i in 1:Nr
        A_mat[i, j] = r[i] * D1[i, j]
    end
    @inbounds for i in 1:Nr
        A_mat[i, i] += one(T)
    end

    if mechanical_bc == :no_slip
        A_mat[idx_inner, :] .= zero(T)
        A_mat[idx_inner, idx_inner] = one(T)
    else
        A_mat[idx_inner, :] .= D1[idx_inner, :]
        A_mat[idx_inner, idx_inner] -= one(T) / r[idx_inner]
    end

    return lu(A_mat)
end

"""
    solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                cd, r_i, r_o, Ra, Pr;
                                mechanical_bc=:no_slip,
                                E=1e-4)

Solve the thermal wind balance equation to compute zonal flow coefficients.

The thermal wind equation in non-dimensional form (viscous time scale) is:

    2Ω̂·∇ū = -Ra E²/Pr × Θ̄ r̂

Taking the φ-component of the curl:

    cos(θ) ∂ū_φ/∂r - sin(θ) ū_φ/r = -(Ra E²)/(2Pr) × (r/r_o) × (1/r) × ∂Θ̄/∂θ

For linear gravity profile g(r) = g_o × r/r_o.

Arguments:
- `uphi_coeffs` : Dict{Int, Vector{T}} - zonal flow coefficients Ū_L(r) (modified in place)
- `duphi_dr_coeffs` : Dict{Int, Vector{T}} - derivatives ∂Ū_L/∂r (modified in place)
- `theta_coeffs` : Dict{Int, Vector{T}} - temperature coefficients Θ̄_ℓ(r)
- `cd` : ChebyshevDiffn - radial discretization
- `r_i, r_o` : inner and outer radii (non-dimensional, typically χ and 1)
- `Ra` : Rayleigh number
- `Pr` : Prandtl number
- `mechanical_bc` : :no_slip or :stress_free
- `E` : Ekman number (CRITICAL - was missing in original!)

Mathematical Details:
--------------------
The θ-derivative of temperature in spectral space:

    ∂Θ̄/∂θ = Σ_ℓ Θ̄_ℓ(r) × ∂Y_ℓ0/∂θ

Using the identity:
    ∂Y_ℓ0/∂θ = -sin(θ) × dP_ℓ/d(cosθ) × √((2ℓ+1)/(4π))

And the recurrence relation:
    sin(θ) dP_ℓ/d(cosθ) = ℓ(ℓ+1)/(2ℓ+1) × [P_{ℓ+1} - P_{ℓ-1}]

We get coupling from temperature mode ℓ to velocity modes L = ℓ±1.

ODE Solver:
-----------
Uses Chebyshev spectral collocation to solve the BVP:
    r dŪ/dr + Ū = f(r)
with boundary condition at the INNER boundary only:
- No-slip: Ū(r_i) = 0
- Stress-free: dŪ/dr - Ū/r = 0 at r_i

NOTE: The thermal wind equation is a first-order ODE, so we can only satisfy
ONE boundary condition. We enforce the inner BC; the outer boundary will have
a small non-zero value consistent with the diagonal approximation.
"""
function solve_thermal_wind_balance!(uphi_coeffs::Dict{Int,Vector{T}},
                            duphi_dr_coeffs::Dict{Int,Vector{T}},
                            theta_coeffs::Dict{Int,Vector{T}},
                            cd,  # ChebyshevDiffn{T}
                            r_i::T, r_o::T, Ra::T, Pr::T;
                            mechanical_bc::Symbol=:no_slip,
                            E::T=T(1e-4)) where T<:Real

    # Validate BC type
    if !(mechanical_bc in (:no_slip, :stress_free))
        error("mechanical_bc must be :no_slip or :stress_free, got: $mechanical_bc")
    end

    r = cd.x
    Nr = length(r)
    D1 = cd.D1

    lmax_theta = maximum(keys(theta_coeffs))

    # Spherical harmonic normalization: Y_ℓ0 = √((2ℓ+1)/(4π)) × P_ℓ(cosθ)
    Y_norm(ℓ::Int) = sqrt(T(2ℓ + 1) / (4 * T(π)))

    # =========================================================================
    # Step 1: Compute forcing coefficients F_L(r) from ∂Θ̄/∂θ
    # =========================================================================
    #
    # The key identity is:
    #   sin(θ) dP_ℓ/d(cosθ) = ℓ(ℓ+1)/(2ℓ+1) × [P_{ℓ+1}(cosθ) - P_{ℓ-1}(cosθ)]
    #
    # Therefore:
    #   ∂Y_ℓ0/∂θ = -ℓ(ℓ+1)/(2ℓ+1) × Y_norm(ℓ) × [P_{ℓ+1}/1 - P_{ℓ-1}/1]
    #
    # Converting P_L back to Y_L0:
    #   ∂Y_ℓ0/∂θ = -ℓ(ℓ+1)/(2ℓ+1) × [Y_norm(ℓ)/Y_norm(ℓ+1) × Y_{ℓ+1,0}
    #                                 - Y_norm(ℓ)/Y_norm(ℓ-1) × Y_{ℓ-1,0}]
    #
    # Projecting ∂Θ̄/∂θ onto Y_L0:
    #   ⟨∂Θ̄/∂θ, Y_L0⟩ = Σ_ℓ Θ̄_ℓ(r) × ⟨∂Y_ℓ0/∂θ, Y_L0⟩
    #
    # Non-zero contributions when L = ℓ±1.

    forcing = Dict{Int, Vector{T}}()

    for (ℓ, θ_coeff) in theta_coeffs
        if ℓ == 0
            continue  # ∂Y_00/∂θ = 0 (uniform temperature has no θ-gradient)
        end

        # Base coupling coefficient from the recurrence relation
        base_coeff = T(ℓ * (ℓ + 1)) / T(2ℓ + 1)

        # Contribution to L = ℓ + 1 mode
        L_plus = ℓ + 1
        norm_ratio_plus = Y_norm(ℓ) / Y_norm(L_plus)
        c_plus = -base_coeff * norm_ratio_plus  # Negative from ∂Y/∂θ formula

        if !haskey(forcing, L_plus)
            forcing[L_plus] = zeros(T, Nr)
        end
        forcing[L_plus] .+= c_plus .* θ_coeff

        # Contribution to L = ℓ - 1 mode (if ℓ ≥ 1)
        if ℓ >= 1
            L_minus = ℓ - 1
            norm_ratio_minus = L_minus == 0 ? Y_norm(ℓ) / Y_norm(0) : Y_norm(ℓ) / Y_norm(L_minus)
            c_minus = base_coeff * norm_ratio_minus  # Positive (double negative)

            if !haskey(forcing, L_minus)
                forcing[L_minus] = zeros(T, Nr)
            end
            forcing[L_minus] .+= c_minus .* θ_coeff
        end
    end

    # =========================================================================
    # Step 2: Compute prefactor with CORRECT scaling
    # =========================================================================
    #
    # Thermal wind equation (non-dimensional with viscous time scale D²/ν):
    #
    #   cos(θ) ∂ū_φ/∂r - sin(θ) ū_φ/r = -(Ra E²)/(2 Pr) × (g(r)/g_o) × (1/r) × ∂Θ̄/∂θ
    #
    # For linear gravity g(r) = g_o × r/r_o:
    #
    #   RHS = -(Ra E²)/(2 Pr r_o) × ∂Θ̄/∂θ
    #
    # IMPORTANT: The E² factor is ESSENTIAL and was missing in the original code!
    # Without E², the zonal flow amplitude is wrong by a factor of E².

    prefactor = -(Ra * E^2) / (2 * Pr * r_o)

    # =========================================================================
    # Step 3: Solve BVP for each L mode using Chebyshev spectral method
    # =========================================================================
    #
    # The thermal wind ODE in the diagonal approximation:
    #   d(r Ū_L)/dr = prefactor × r² × F_L(r)
    #
    # Rewritten as:
    #   r dŪ/dr + Ū = f(r)   where f(r) = prefactor × r² × F_L(r)
    #
    # In matrix form: (diag(r) @ D1 + I) @ Ū = f
    #
    # Boundary conditions:
    # - No-slip: Ū(r_boundary) = 0
    # - Stress-free: dŪ/dr - Ū/r = 0  ⟺  (D1 - diag(1/r)) @ Ū = 0 at boundary

    # Determine boundary indices (depends on grid ordering)
    # Chebyshev grids typically have r[1] at one boundary, r[Nr] at the other
    # We need to identify which is inner (r_i) and which is outer (r_o)
    idx_inner = abs(r[1] - r_i) < abs(r[Nr] - r_i) ? 1 : Nr
    r2 = r .^ 2
    tw_lu = _thermal_wind_operator_lu(r, D1, idx_inner, mechanical_bc)

    for (L, F_L) in forcing
        # RHS for the ODE: r dŪ/dr + Ū = f(r)
        f_rhs = prefactor .* r2 .* F_L
        f_rhs[idx_inner] = zero(T)

        # Solve the linear system
        uphi_L = tw_lu \ f_rhs

        # Store results
        uphi_coeffs[L] = uphi_L

        # Compute radial derivative using Chebyshev differentiation
        duphi_dr_coeffs[L] = D1 * uphi_L
    end

    # =========================================================================
    # Step 4: Zero out modes that have no forcing
    # =========================================================================

    forced_modes = Set(keys(forcing))
    for ℓ in keys(theta_coeffs)
        if !haskey(uphi_coeffs, ℓ) || !(ℓ in forced_modes)
            uphi_coeffs[ℓ] = zeros(T, Nr)
            duphi_dr_coeffs[ℓ] = zeros(T, Nr)
        end
    end

    return nothing
end


"""
    solve_thermal_wind_balance_3d!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs, m_bs,
                                   cd, r_i, r_o, Ra, Pr;
                                   mechanical_bc=:no_slip, E=1e-4)

Solve thermal wind balance for a specific azimuthal wavenumber m_bs.

This extends the axisymmetric thermal wind solver to handle non-axisymmetric
temperature variations Y_ℓm with m ≠ 0.

The key difference from m=0 is that the spherical harmonic coupling coefficients
depend on m through the associated Legendre functions:

    ∂Y_ℓm/∂θ = m cot(θ) Y_ℓm + √[(ℓ-m)(ℓ+m+1)] Y_{ℓ,m+1} e^{-iφ}  (complex form)

For real spherical harmonics with fixed m:
    ∂Y_ℓm/∂θ couples to Y_{ℓ±1,m}

The coupling coefficients are:
    c_{ℓ→ℓ+1,m} = √[(ℓ+1)² - m²] / (2ℓ+1) × (ℓ+1)
    c_{ℓ→ℓ-1,m} = √[ℓ² - m²] / (2ℓ+1) × ℓ

Arguments:
- `uphi_coeffs` : velocity coefficients for mode m (modified in place)
- `duphi_dr_coeffs` : velocity derivatives (modified in place)
- `theta_coeffs` : temperature coefficients {ℓ => θ̄_ℓm(r)} for fixed m
- `m_bs` : azimuthal wavenumber of the basic state
- `cd` : Chebyshev differentiation
- `r_i, r_o` : radii
- `Ra, Pr, E` : physical parameters
- `mechanical_bc` : boundary condition type
"""
function solve_thermal_wind_balance_3d!(uphi_coeffs::Dict{Int,Vector{T}},
                                        duphi_dr_coeffs::Dict{Int,Vector{T}},
                                        theta_coeffs::Dict{Int,Vector{T}},
                                        m_bs::Int,
                                        cd,
                                        r_i::T, r_o::T, Ra::T, Pr::T;
                                        mechanical_bc::Symbol=:no_slip,
                                        E::T=T(1e-4)) where T<:Real

    # For m=0, delegate to the standard solver
    if m_bs == 0
        solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                    cd, r_i, r_o, Ra, Pr;
                                    mechanical_bc=mechanical_bc, E=E)
        return nothing
    end

    # Validate BC type
    if !(mechanical_bc in (:no_slip, :stress_free))
        error("mechanical_bc must be :no_slip or :stress_free, got: $mechanical_bc")
    end

    r = cd.x
    Nr = length(r)
    D1 = cd.D1

    lmax_theta = isempty(theta_coeffs) ? 0 : maximum(keys(theta_coeffs))

    # Spherical harmonic normalization for Y_ℓm
    # For m≠0: Y_ℓm includes factor √(2) for real spherical harmonics
    function Y_norm_m(ℓ::Int, m::Int)
        if m == 0
            return sqrt(T(2ℓ + 1) / (4 * T(π)))
        else
            return sqrt(T(2) * T(2ℓ + 1) / (4 * T(π)))
        end
    end

    # =========================================================================
    # Compute forcing coefficients F_L(r) from ∂Θ̄/∂θ for fixed m
    # =========================================================================
    #
    # The θ-derivative of spherical harmonics Y_ℓm couples to Y_{ℓ±1,m}:
    #
    #   ∂Y_ℓm/∂θ = ℓ cot(θ) Y_ℓm - √[(ℓ+m)(ℓ-m+1)/(2ℓ+1)(2ℓ-1)] × (2ℓ+1) Y_{ℓ-1,m}
    #            + √[(ℓ-m)(ℓ+m+1)/(2ℓ+1)(2ℓ+3)] × (2ℓ+1) Y_{ℓ+1,m}
    #
    # Using the recurrence for associated Legendre functions:
    #   (1-x²) dP_ℓ^m/dx = -ℓx P_ℓ^m + (ℓ+m) P_{ℓ-1}^m
    #                    = (ℓ+1)x P_ℓ^m - (ℓ-m+1) P_{ℓ+1}^m
    #
    # For the thermal wind, we need sin(θ)⁻¹ × ∂Θ̄/∂θ, which projects as:
    #   ⟨sin(θ)⁻¹ ∂Y_ℓm/∂θ, Y_Lm⟩ gives coupling coefficients
    #
    # Simplified coupling (following Kore/standard approach):
    #   Temperature ℓ,m → Velocity L=ℓ-1,m: c_{-} = √[(ℓ²-m²)/(4ℓ²-1)] × ℓ
    #   Temperature ℓ,m → Velocity L=ℓ+1,m: c_{+} = √[((ℓ+1)²-m²)/((2ℓ+1)(2ℓ+3))] × (ℓ+1)

    forcing = Dict{Int, Vector{T}}()

    for (ℓ, θ_coeff) in theta_coeffs
        if ℓ < m_bs
            continue  # Invalid: ℓ must be ≥ m
        end

        if maximum(abs, θ_coeff) < 1e-15
            continue  # Skip negligible modes
        end

        # Coupling to L = ℓ - 1 (if ℓ > m, so that L ≥ m)
        if ℓ > m_bs
            L_minus = ℓ - 1
            # c_{-} = √[(ℓ²-m²)/(4ℓ²-1)] × ℓ × norm_ratio
            denom_minus = T(4 * ℓ^2 - 1)
            if denom_minus > 0
                c_minus = sqrt(T(ℓ^2 - m_bs^2) / denom_minus) * T(ℓ)
                norm_ratio = Y_norm_m(ℓ, m_bs) / Y_norm_m(L_minus, m_bs)
                c_minus *= norm_ratio

                if !haskey(forcing, L_minus)
                    forcing[L_minus] = zeros(T, Nr)
                end
                forcing[L_minus] .+= c_minus .* θ_coeff
            end
        end

        # Coupling to L = ℓ + 1 (always valid)
        L_plus = ℓ + 1
        # c_{+} = -√[((ℓ+1)²-m²)/((2ℓ+1)(2ℓ+3))] × (ℓ+1) × norm_ratio
        # Note: negative sign comes from the derivative relation
        denom_plus = T((2ℓ + 1) * (2ℓ + 3))
        numer_plus = T((ℓ + 1)^2 - m_bs^2)
        if numer_plus > 0 && denom_plus > 0
            c_plus = -sqrt(numer_plus / denom_plus) * T(ℓ + 1)
            norm_ratio_plus = Y_norm_m(ℓ, m_bs) / Y_norm_m(L_plus, m_bs)
            c_plus *= norm_ratio_plus

            if !haskey(forcing, L_plus)
                forcing[L_plus] = zeros(T, Nr)
            end
            forcing[L_plus] .+= c_plus .* θ_coeff
        end
    end

    # =========================================================================
    # Prefactor with E² scaling (same as axisymmetric case)
    # =========================================================================
    prefactor = -(Ra * E^2) / (2 * Pr * r_o)

    # =========================================================================
    # Solve BVP for each L mode using Chebyshev spectral method
    # =========================================================================
    #
    # The thermal wind ODE in the diagonal approximation:
    #   d(r Ū_L)/dr = prefactor × r² × F_L(r)
    #
    # Rewritten as:
    #   r dŪ/dr + Ū = f(r)   where f(r) = prefactor × r² × F_L(r)
    #
    # In matrix form: (diag(r) @ D1 + I) @ Ū = f
    #
    # Boundary condition (INNER only - first-order ODE can only satisfy one BC):
    # - No-slip: Ū(r_i) = 0
    # - Stress-free: dŪ/dr - Ū/r = 0 at r_i

    # Determine boundary indices
    idx_inner = abs(r[1] - r_i) < abs(r[Nr] - r_i) ? 1 : Nr
    r2 = r .^ 2
    tw_lu = _thermal_wind_operator_lu(r, D1, idx_inner, mechanical_bc)

    for (L, F_L) in forcing
        if L < m_bs
            continue  # L must be ≥ m
        end

        # RHS for the ODE: r dŪ/dr + Ū = f(r)
        f_rhs = prefactor .* r2 .* F_L
        f_rhs[idx_inner] = zero(T)

        # Solve the linear system
        uphi_L = tw_lu \ f_rhs

        # Store results
        uphi_coeffs[L] = uphi_L
        duphi_dr_coeffs[L] = D1 * uphi_L
    end

    # Zero out modes without forcing
    forced_modes = Set(keys(forcing))
    for ℓ in keys(theta_coeffs)
        if !haskey(uphi_coeffs, ℓ) || !(ℓ in forced_modes)
            uphi_coeffs[ℓ] = zeros(T, Nr)
            duphi_dr_coeffs[ℓ] = zeros(T, Nr)
        end
    end

    return nothing
end


# =============================================================================
#  Full Coupled Thermal Wind Solver (No Diagonal Approximation)
#
#  Solves the thermal wind equation without the diagonal approximation by
#  treating the full mode coupling from the (ẑ·∇) operator on the LHS.
#
#  The equation is:
#    cos(θ) ∂ū_φ/∂r - (sin(θ)/r) ∂ū_φ/∂θ = -(Ra E²)/(2 Pr r_o) × ∂Θ̄/∂θ
#
#  The cos(θ) and sin(θ)∂/∂θ terms couple mode L to modes L±1, creating a
#  tridiagonal system of coupled ODEs that must be solved simultaneously.
# =============================================================================

"""
    _orthonormal_plm(lmax, m, x) -> Vector

Orthonormal associated Legendre functions P̄_ℓ^m(x) for ℓ = 0..lmax, normalized
so that ∫_{-1}^1 (P̄_ℓ^m)² dx = 1 (the θ-part of the sphere-orthonormal Y_ℓ^m).
"""
function _orthonormal_plm(lmax::Int, m::Int, x::T) where {T<:Real}
    P = zeros(T, lmax + 1)
    m > lmax && return P
    somx2 = sqrt((one(T) - x) * (one(T) + x))
    # unnormalized P_m^m = (-1)^m (2m-1)!! (1-x²)^{m/2}
    pmm = one(T)
    fact = one(T)
    for _ in 1:m
        pmm *= -fact * somx2
        fact += T(2)
    end
    # orthonormal-in-x factor c_ℓ = √[(2ℓ+1)/2 · (ℓ-m)!/(ℓ+m)!]
    cfac = function (l)
        ratio = one(T)
        for k in (l - m + 1):(l + m)
            ratio /= T(k)
        end
        return sqrt(T(2l + 1) / T(2) * ratio)
    end
    P[m + 1] = cfac(m) * pmm
    pl1 = x * T(2m + 1) * pmm                # unnormalized P_{m+1}^m
    (m + 1 <= lmax) && (P[m + 2] = cfac(m + 1) * pl1)
    pl2 = pmm
    for l in (m + 2):lmax
        pl = (x * T(2l - 1) * pl1 - T(l + m - 1) * pl2) / T(l - m)
        P[l + 1] = cfac(l) * pl
        pl2 = pl1
        pl1 = pl
    end
    return P
end

"""
    _dtheta_sphere_projection(Kset, Lset, m, T) -> Dict{Tuple{Int,Int},T}

Projection coefficients M[(K,ℓ)] = ⟨∂Y_ℓ^m/∂θ, Y_K^m⟩ over the unit sphere,
for velocity modes K ∈ `Kset` and temperature modes ℓ ∈ `Lset`. Uses the exact
identity sin(θ)∂_θY_ℓ = ℓα_ℓ⁺Y_{ℓ+1} - (ℓ+1)α_ℓ⁻Y_{ℓ-1} together with
Gauss–Chebyshev quadrature of G(a,b) = ∫ P̄_a^m P̄_b^m / √(1-x²) dx. Unlike the
diagonal-approximation coupling, this retains the full ℓ±1, ℓ±3, … structure
required to be consistent with the coupled solver's (orthonormal) cosθ/sinθ∂θ
operators.
"""
function _dtheta_sphere_projection(Kset, Lset, m::Int, ::Type{T}) where {T<:Real}
    (isempty(Kset) || isempty(Lset)) && return Dict{Tuple{Int,Int},T}()
    lmax_needed = max(maximum(Kset), maximum(Lset) + 1)
    Nq = lmax_needed + 2
    Ptab = Matrix{T}(undef, Nq, lmax_needed + 1)
    for j in 1:Nq
        x = cos(T(π) * (T(j) - T(0.5)) / T(Nq))
        Ptab[j, :] .= _orthonormal_plm(lmax_needed, m, x)
    end
    Gw = T(π) / T(Nq)
    Gint = function (a, b)
        (a < m || b < m || a > lmax_needed || b > lmax_needed) && return zero(T)
        s = zero(T)
        @inbounds for j in 1:Nq
            s += Ptab[j, a + 1] * Ptab[j, b + 1]
        end
        return Gw * s
    end
    αplus(l)  = (l + 1)^2 - m^2 > 0 ? sqrt(T((l + 1)^2 - m^2) / T((2l + 1) * (2l + 3))) : zero(T)
    αminus(l) = l^2 - m^2 > 0 ? sqrt(T(l^2 - m^2) / T((2l - 1) * (2l + 1))) : zero(T)
    M = Dict{Tuple{Int,Int},T}()
    for ℓ in Lset
        ℓ < m && continue
        for K in Kset
            val = T(ℓ) * αplus(ℓ) * Gint(K, ℓ + 1)
            if ℓ - 1 >= m
                val -= T(ℓ + 1) * αminus(ℓ) * Gint(K, ℓ - 1)
            end
            abs(val) > eps(T) && (M[(K, ℓ)] = val)
        end
    end
    return M
end

"""
    solve_thermal_wind_coupled!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                m_bs, cd, r_i, r_o, Ra, Pr;
                                mechanical_bc=:no_slip, E=1e-4, lmax=nothing)

Solve thermal wind balance WITHOUT the diagonal approximation.

This solver accounts for the full mode coupling from the (ẑ·∇) operator:
- cos(θ) couples Y_Lm → Y_{L±1,m}
- sin(θ) ∂/∂θ also couples Y_Lm → Y_{L±1,m}

The result is a coupled tridiagonal system of ODEs in L that is solved
simultaneously using a spectral Chebyshev discretization.

# Arguments
- `uphi_coeffs` : Output dictionary for velocity coefficients (modified in place)
- `duphi_dr_coeffs` : Output dictionary for velocity derivatives (modified in place)
- `theta_coeffs` : Input temperature coefficients {ℓ => θ̄_ℓm(r)}
- `m_bs` : Azimuthal wavenumber
- `cd` : ChebyshevDiffn structure
- `r_i, r_o` : Inner and outer radii
- `Ra, Pr, E` : Rayleigh, Prandtl, Ekman numbers
- `mechanical_bc` : :no_slip or :stress_free
- `lmax` : Maximum L for velocity (defaults to max key in theta_coeffs + 1)
"""
function solve_thermal_wind_coupled!(uphi_coeffs::Dict{Int,Vector{T}},
                                     duphi_dr_coeffs::Dict{Int,Vector{T}},
                                     theta_coeffs::Dict{Int,Vector{T}},
                                     m_bs::Int,
                                     cd,
                                     r_i::T, r_o::T, Ra::T, Pr::T;
                                     mechanical_bc::Symbol=:no_slip,
                                     E::T=T(1e-4),
                                     lmax::Union{Nothing,Int}=nothing) where T<:Real

    # NOTE: m=0 is handled by the SAME coupled operator below — it is deliberately
    # NOT delegated to the diagonal solve_thermal_wind_balance! heuristic. That
    # heuristic puts the axisymmetric zonal flow at the wrong spherical-harmonic
    # parity (odd L for a Y_ℓ0-even forcing) and does not satisfy the thermal-wind
    # PDE (the true cosθ/sinθ∂θ operator is zero-diagonal). Velocity modes start at
    # L = max(m_bs, 1): u_φ has no ℓ=0 component (a constant-in-θ zonal flow is
    # unphysical and pole-singular under advection).

    if !(mechanical_bc in (:no_slip, :stress_free))
        error("mechanical_bc must be :no_slip or :stress_free, got: $mechanical_bc")
    end

    r = cd.x
    Nr = length(r)
    D1 = cd.D1

    # Determine lmax from temperature coefficients
    lmax_theta = isempty(theta_coeffs) ? m_bs : maximum(keys(theta_coeffs))
    lmax_vel = lmax === nothing ? lmax_theta + 1 : lmax
    lmax_vel = max(lmax_vel, m_bs)  # L ≥ m required
    Lmin = max(m_bs, 1)             # exclude unphysical ℓ=0 zonal flow; for m≥1, Lmin=m_bs

    # Number of velocity modes: L = Lmin, Lmin+1, ..., lmax_vel
    n_modes = lmax_vel - Lmin + 1
    if n_modes < 1
        return nothing
    end

    # The cosθ coupling matrix A (built below) is zero-diagonal tridiagonal, which
    # is singular for an ODD number of modes (det recurrence d_n = -a·b·d_{n-2},
    # d_1 = 0 ⇒ det = 0 for every odd dim). That makes the coupled thermal-wind BVP
    # singular (e.g. m_bs=1 with lmax_vel=9). Pad by one degree so n_modes is even
    # and A is non-singular; the extra high-degree mode is a small truncation
    # correction.
    if isodd(n_modes)
        lmax_vel += 1
        n_modes += 1
    end

    # Mode indices: mode_idx[k] = L means the k-th mode corresponds to degree L
    mode_idx = collect(Lmin:lmax_vel)

    # =========================================================================
    # Coupling coefficients
    # =========================================================================
    #
    # cos(θ) Y_Lm = α_L^+ Y_{L+1,m} + α_L^- Y_{L-1,m}
    # where:
    #   α_L^+ = √[((L+1)²-m²)/((2L+1)(2L+3))]
    #   α_L^- = √[(L²-m²)/((2L-1)(2L+1))]
    #
    # sin(θ) ∂Y_Lm/∂θ (verified numerically against orthonormal Y_lm):
    #   sin(θ) ∂Y_Lm/∂θ = L α_L^+ Y_{L+1,m} - (L+1) α_L^- Y_{L-1,m}

    function alpha_plus(L::Int)
        num = (L + 1)^2 - m_bs^2
        den = (2L + 1) * (2L + 3)
        return num > 0 && den > 0 ? sqrt(T(num) / T(den)) : zero(T)
    end

    function alpha_minus(L::Int)
        num = L^2 - m_bs^2
        den = (2L - 1) * (2L + 1)
        return num > 0 && den > 0 ? sqrt(T(num) / T(den)) : zero(T)
    end

    # =========================================================================
    # Build the coupling matrices A and B
    # =========================================================================
    #
    # The thermal wind equation projected onto Y_Km:
    #   Σ_L [A_{KL} dŪ_L/dr - (1/r) B_{KL} Ū_L] = F_K(r)
    #
    # where:
    #   A_{KL} = ⟨cos(θ) Y_Lm, Y_Km⟩
    #   B_{KL} = ⟨sin(θ) ∂Y_Lm/∂θ, Y_Km⟩

    A = zeros(T, n_modes, n_modes)  # cos(θ) coupling
    B = zeros(T, n_modes, n_modes)  # sin(θ)∂/∂θ coupling

    for (k, K) in enumerate(mode_idx)
        # Contribution from L = K-1 (if valid)
        if K > m_bs
            l = K - 1
            j = findfirst(==(l), mode_idx)
            if j !== nothing
                # A_{K,L} where L = K-1: ⟨cos(θ) Y_{K-1,m}, Y_Km⟩ = α_{K-1}^+
                A[k, j] = alpha_plus(l)

                # B_{K,L} where L = K-1:
                # From: sin(θ)∂Y_{K-1,m}/∂θ = (K-1) α_{K-1}^+ Y_{K,m} - K α_{K-1}^- Y_{K-2,m}
                # Projection onto Y_Km: ⟨sin(θ)∂Y_{K-1,m}/∂θ, Y_Km⟩ = (K-1) α_{K-1}^+
                B[k, j] = l * alpha_plus(l)  # = (K-1) α_{K-1}^+
            end
        end

        # Contribution from L = K+1 (if valid)
        if K < lmax_vel
            l = K + 1
            j = findfirst(==(l), mode_idx)
            if j !== nothing
                # A_{K,L} where L = K+1: ⟨cos(θ) Y_{K+1,m}, Y_Km⟩ = α_{K+1}^-
                A[k, j] = alpha_minus(l)

                # B_{K,L} where L = K+1:
                # From: sin(θ)∂Y_{K+1,m}/∂θ = (K+1) α_{K+1}^+ Y_{K+2,m} - (K+2) α_{K+1}^- Y_{K,m}
                # Projection onto Y_Km: ⟨sin(θ)∂Y_{K+1,m}/∂θ, Y_Km⟩ = -(K+2) α_{K+1}^-
                B[k, j] = -(l + 1) * alpha_minus(l)  # = -(K+2) α_{K+1}^-
            end
        end
    end

    # =========================================================================
    # Compute forcing F_K(r) from temperature gradient
    # =========================================================================
    #
    # F_K(r) = -(Ra E²)/(2 Pr r_o) × ⟨∂Θ̄/∂θ, Y_Km⟩
    #
    # The forcing is the full sphere projection ⟨∂Y_ℓm/∂θ, Y_Km⟩, which couples
    # ℓ → K = ℓ±1, ℓ±3, …. This matches the orthonormal cosθ/sinθ∂θ operators
    # assembled above (verified by the manufactured-solution test). The diagonal
    # ℓ±1-only approximation (theta_derivative_coeff_3d) is NOT used here because
    # it is inconsistent with the coupled operators.

    prefactor = -(Ra * E^2) / (2 * Pr * r_o)

    F = zeros(T, n_modes, Nr)  # F[k, i] = F_{mode_idx[k]}(r[i])

    Lset = Int[ℓ for (ℓ, θ_coeff) in theta_coeffs
               if ℓ >= m_bs && maximum(abs, θ_coeff) >= 1e-15]
    if !isempty(Lset)
        Mproj = _dtheta_sphere_projection(mode_idx, Lset, m_bs, T)
        for ℓ in Lset
            θ_coeff = theta_coeffs[ℓ]
            for (k, K) in enumerate(mode_idx)
                c = get(Mproj, (K, ℓ), zero(T))
                c == zero(T) && continue
                F[k, :] .+= (prefactor * c) .* θ_coeff
            end
        end
    end

    # =========================================================================
    # Build and solve the full linear system
    # =========================================================================
    #
    # The equation A × dŪ/dr - (1/r) × B × Ū = F becomes:
    # (A ⊗ D1 - diag(1/r) × B ⊗ I) × vec(Ū) = vec(F)
    #
    # where vec(Ū) stacks all modes: [Ū_{m}; Ū_{m+1}; ...; Ū_{lmax}]
    # and each Ū_L has Nr components.

    # Total unknowns: n_modes × Nr
    n_total = n_modes * Nr

    # Build the system matrix
    # Index convention: u[(k-1)*Nr + i] = Ū_{mode_idx[k]}(r[i])
    L_op = zeros(T, n_total, n_total)

    for k1 in 1:n_modes  # row (equation for mode K = mode_idx[k1])
        for k2 in 1:n_modes  # column (contribution from mode L = mode_idx[k2])
            # Block (k1, k2) has size Nr × Nr
            row_start = (k1 - 1) * Nr + 1
            row_end = k1 * Nr
            col_start = (k2 - 1) * Nr + 1
            col_end = k2 * Nr

            if abs(A[k1, k2]) > 1e-15
                # A_{k1,k2} × D1 contribution
                coeff = A[k1, k2]
                @inbounds for j in 1:Nr
                    col = col_start + j - 1
                    for i in 1:Nr
                        L_op[row_start + i - 1, col] += coeff * D1[i, j]
                    end
                end
            end

            if abs(B[k1, k2]) > 1e-15
                # -(1/r) × B_{k1,k2} × I contribution
                for i in 1:Nr
                    L_op[row_start + i - 1, col_start + i - 1] -= B[k1, k2] / r[i]
                end
            end
        end
    end

    # RHS vector
    F_vec = vec(F')  # Flatten: [F[1,:]; F[2,:]; ...]

    # =========================================================================
    # Apply boundary conditions
    # =========================================================================
    #
    # For each mode K, apply one BC at the inner boundary:
    # - No-slip: Ū_K(r_i) = 0
    # - Stress-free: dŪ_K/dr - Ū_K/r = 0
    #
    # The coupled thermal-wind balance is still first-order in radius. Applying
    # both radial boundaries would replace two collocation equations per mode and
    # overconstrain the continuous problem.

    idx_inner = abs(r[1] - r_i) < abs(r[Nr] - r_i) ? 1 : Nr

    for k in 1:n_modes
        # Row index for the BC at inner boundary for mode k
        bc_row = (k - 1) * Nr + idx_inner

        if mechanical_bc == :no_slip
            # Dirichlet: Ū = 0 at inner boundary
            L_op[bc_row, :] .= zero(T)
            L_op[bc_row, bc_row] = one(T)
            F_vec[bc_row] = zero(T)
        else  # stress_free
            # Robin: dŪ/dr - Ū/r = 0 at inner boundary
            L_op[bc_row, :] .= zero(T)
            col_start = (k - 1) * Nr + 1
            col_end = k * Nr
            L_op[bc_row, col_start:col_end] .= D1[idx_inner, :]
            L_op[bc_row, bc_row] -= one(T) / r[idx_inner]
            F_vec[bc_row] = zero(T)
        end
    end

    # =========================================================================
    # Solve the system
    # =========================================================================
    #
    # The discretized first-order coupled BVP has a condition number that grows
    # exponentially with lmax (the cosθ/sinθ∂θ transport couples modes with
    # growth rate ∝ L/r, so the fundamental solution spans ~exp(L·Δr)). Above
    # lmax≈16-18 (Float64) cond(L_op) exceeds machine precision and the solve is
    # unreliable. Estimate the reciprocal condition number and warn rather than
    # silently return garbage. A robust high-lmax solve would need a stiff-BVP
    # method (e.g. marching with re-orthonormalization); basic states are
    # low-degree in practice, so capping lmax is the pragmatic remedy.

    F_lu = lu(L_op; check=false)
    rcond = issuccess(F_lu) ?
        LinearAlgebra.LAPACK.gecon!('1', F_lu.factors, opnorm(L_op, 1)) : zero(real(T))
    if rcond < eps(real(T))
        # Numerically singular. The thermal-wind balance determines ū_φ only up
        # to geostrophic modes u_φ = f(r sinθ); for stress-free with m_bs = 1 the
        # mode ū_φ ∝ r·Y_1^1 (= r sinθ) satisfies the interior equations AND the
        # Robin boundary condition exactly, so the discrete system has an exact
        # null vector (higher truncations approach singularity through
        # near-geostrophic modes). An LU solve would return an arbitrary
        # null-space component; take the minimum-norm least-squares solution
        # instead — deterministic and free of the undetermined geostrophic part.
        U_vec = pinv(L_op) * F_vec
        @warn "Coupled thermal-wind system is singular (zonal flow determined \
               only up to a geostrophic mode); returning the minimum-norm \
               solution." rcond lmax_vel n_modes maxlog=1
    else
        U_vec = F_lu \ F_vec
    end

    # =========================================================================
    # Extract results
    # =========================================================================

    for (k, L) in enumerate(mode_idx)
        start_idx = (k - 1) * Nr + 1
        end_idx = k * Nr
        uphi_coeffs[L] = U_vec[start_idx:end_idx]
        duphi_dr_coeffs[L] = D1 * uphi_coeffs[L]
    end

    # Zero out modes without forcing
    for ℓ in keys(theta_coeffs)
        if !haskey(uphi_coeffs, ℓ)
            uphi_coeffs[ℓ] = zeros(T, Nr)
            duphi_dr_coeffs[ℓ] = zeros(T, Nr)
        end
    end

    return nothing
end

"""
    theta_derivative_coeff_3d(ℓ::Int, m::Int)

Compute θ-derivative coupling coefficients for ⟨∂Θ/∂θ, Y_Lm⟩ forcing.

Returns (c_plus, c_minus), including the spherical-harmonic normalization ratio
N_ℓ/N_L (4π-normalized Y, where N_ℓ = √((2ℓ+1)/4π)), so the coefficients are
consistent with `solve_thermal_wind_balance_3d!` (the analytically-validated
diagonal solver) and the test helper `coupling_coeff_plus_3d`:

- c_plus  = -(ℓ+1) × √[((ℓ+1)²-m²)/((2ℓ+1)(2ℓ+3))] × √((2ℓ+1)/(2ℓ+3))   (Y_ℓm → Y_{ℓ+1,m})
- c_minus = +ℓ     × √[(ℓ²-m²)/((2ℓ-1)(2ℓ+1))]     × √((2ℓ+1)/(2ℓ-1))   (Y_ℓm → Y_{ℓ-1,m})

The √2 from real-harmonic normalization (m≠0) cancels in the ratio, so the same
factor applies for all m.

# Example
```julia
c_plus, c_minus = theta_derivative_coeff_3d(2, 1)  # For Y_21
# c_plus ≈ -1.2127 (coupling to Y_31)
# c_minus ≈ 1.1547 (coupling to Y_11)
```
"""
function theta_derivative_coeff_3d(ℓ::Int, m::Int)
    if ℓ < abs(m)
        return (0.0, 0.0)
    end

    c_plus = 0.0
    c_minus = 0.0

    # Coupling to ℓ+1 (includes norm ratio N_ℓ/N_{ℓ+1} = √((2ℓ+1)/(2ℓ+3)))
    if ℓ >= 0
        num_plus = (ℓ + 1)^2 - m^2
        den_plus = (2*ℓ + 1) * (2*ℓ + 3)
        if num_plus >= 0 && den_plus > 0
            c_plus = -(ℓ + 1) * sqrt(num_plus / den_plus) *
                     sqrt((2*ℓ + 1) / (2*ℓ + 3))
        end
    end

    # Coupling to ℓ-1 (includes norm ratio N_ℓ/N_{ℓ-1} = √((2ℓ+1)/(2ℓ-1)))
    if ℓ > abs(m)
        num_minus = ℓ^2 - m^2
        den_minus = (2*ℓ - 1) * (2*ℓ + 1)
        if num_minus >= 0 && den_minus > 0
            c_minus = ℓ * sqrt(num_minus / den_minus) *
                      sqrt((2*ℓ + 1) / (2*ℓ - 1))
        end
    end

    return (c_plus, c_minus)
end
