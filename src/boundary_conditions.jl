# =============================================================================
#  Boundary-condition utilities in toroidal–poloidal representation
#
#  This module contains boundary condition implementations for:
#  - Mechanical (velocity) boundary conditions
#  - Thermal boundary conditions
#  - Magnetic field boundary conditions
# =============================================================================

using SparseArrays: SparseMatrixCSC
using SpecialFunctions: sphericalbesselj

"""
    velocity_from_potentials(op, P, T)

Convert poloidal (`P`) and toroidal (`T`) potentials defined on the operator
collocation grid into velocity components `(u_r, u_θ, u_φ)` for the azimuthal
wavenumber `op.params.m`.

The formulas follow the standard decomposition

```
    u = ∇ × ∇ × (P r̂) + ∇ × (T r̂)
```

assuming fields vary as `exp(i m φ)`.  The returned arrays share the same shape
as the input potentials.
"""
function velocity_from_potentials(op, P, T)
    Nr, Nθ = size(P)
    size(T) == size(P) || throw(DimensionMismatch("P and T must have same size"))
    size(op.Dr, 1) == Nr || throw(DimensionMismatch("Dr must have $Nr rows"))
    size(op.Dr, 2) == Nr || throw(DimensionMismatch("Dr must have $Nr columns"))
    size(op.Dθ, 1) == Nθ || throw(DimensionMismatch("Dθ must have $Nθ rows"))
    size(op.Dθ, 2) == Nθ || throw(DimensionMismatch("Dθ must have $Nθ columns"))
    size(op.Lθ, 1) == Nθ || throw(DimensionMismatch("Lθ must have $Nθ rows"))
    size(op.Lθ, 2) == Nθ || throw(DimensionMismatch("Lθ must have $Nθ columns"))

    # Angular derivatives
    dθ_T = T * op.Dθ'
    lap_ang_P = P * op.Lθ'

    # Radial derivatives of the potentials
    dr_P = op.Dr * P

    # Common geometric factors
    inv_r = _get_inv_r(op, Nr)
    inv_r2 = inv_r .* inv_r

    im_m = _get_im_m(op)

    # Velocity components
    u_r = -lap_ang_P .* inv_r2
    u_θ = (dr_P * op.Dθ') .* inv_r
    u_φ = -(dθ_T .* inv_r)

    if !iszero(im_m)
        inv_r_sinθ = _get_inv_r_sinθ(op, inv_r, Nr, Nθ)
        u_θ .+= (im_m .* T) .* inv_r_sinθ
        u_φ .+= (im_m .* dr_P) .* inv_r_sinθ
    end

    return u_r, u_θ, u_φ
end

"""
    apply_mechanical_bc_from_potentials!(res_r, res_θ, res_φ,
                                         P, T, op;
                                         inner::Symbol=:no_slip,
                                         outer::Symbol=:no_slip)

Overwrite the boundary rows of the residual blocks `(res_r, res_θ, res_φ)` using
velocity boundary conditions derived from the toroidal–poloidal potentials
`(P, T)`.

Supported mechanical boundary types:

- `:no_slip`      → `u_r = u_θ = u_φ = 0`
- `:stress_free`  → `u_r = 0`, `∂_r u_θ = u_θ / r`, `∂_r u_φ = u_φ / r`

The function evaluates the necessary velocity components (and their radial
derivatives) internally from the potentials.
"""
function apply_mechanical_bc_from_potentials!(res_r, res_θ, res_φ,
                                              P, T, op;
                                              inner::Symbol=:no_slip,
                                              outer::Symbol=:no_slip)
    size(T) == size(P) || throw(DimensionMismatch("P and T must have same size"))
    size(res_r) == size(P) || throw(DimensionMismatch("res_r must match P size"))
    size(res_θ) == size(P) || throw(DimensionMismatch("res_θ must match P size"))
    size(res_φ) == size(P) || throw(DimensionMismatch("res_φ must match P size"))

    u_r, u_θ, u_φ = velocity_from_potentials(op, P, T)
    dr_uθ = op.Dr * u_θ
    dr_uφ = op.Dr * u_φ

    Nr = size(P, 1)
    inner_idx, outer_idx = _boundary_indices(op, Nr)
    inv_r = _get_inv_r(op, Nr)

    enforce_mechanical_bc_at!(res_r, res_θ, res_φ,
                              u_r, u_θ, u_φ,
                              dr_uθ, dr_uφ,
                              inv_r, inner, inner_idx)

    enforce_mechanical_bc_at!(res_r, res_θ, res_φ,
                              u_r, u_θ, u_φ,
                              dr_uθ, dr_uφ,
                              inv_r, outer, outer_idx)
    return nothing
end

function enforce_mechanical_bc_at!(res_r, res_θ, res_φ,
                                   u_r, u_θ, u_φ,
                                   dr_uθ, dr_uφ,
                                   inv_r, bc::Symbol, idx::Int)
    if bc === :no_slip
        res_r[idx, :] .= u_r[idx, :]
        res_θ[idx, :] .= u_θ[idx, :]
        res_φ[idx, :] .= u_φ[idx, :]
    elseif bc === :stress_free
        inv_r_val = _inv_r_at(inv_r, idx)
        res_r[idx, :] .= u_r[idx, :]
        res_θ[idx, :] .= dr_uθ[idx, :] .- u_θ[idx, :] .* inv_r_val
        res_φ[idx, :] .= dr_uφ[idx, :] .- u_φ[idx, :] .* inv_r_val
    else
        throw(ArgumentError("Unsupported mechanical boundary condition: $(bc)"))
    end
end

"""
    apply_thermal_bc_from_potentials!(res_T, Θ, op;
                                      inner::Symbol=:fixed_temperature,
                                      outer::Symbol=:fixed_temperature,
                                      value_inner::Real=0.0,
                                      value_outer::Real=0.0,
                                      flux_inner::Real=0.0,
                                      flux_outer::Real=0.0)

Apply thermal boundary conditions directly to the temperature residual block
`res_T`.  The helper mirrors the mechanical routine but does not require
potentials explicitly; it is defined here so that a single module hosts all
boundary utilities for the toroidal–poloidal formulation.

Supported thermal boundary types:

- `:fixed_temperature` → Θ = prescribed value
- `:fixed_flux`        → ∂_r Θ = prescribed flux
"""
function apply_thermal_bc_from_potentials!(res_T, Θ, op;
                                           inner::Symbol=:fixed_temperature,
                                           outer::Symbol=:fixed_temperature,
                                           value_inner::Real=0.0,
                                           value_outer::Real=0.0,
                                           flux_inner::Real=0.0,
                                           flux_outer::Real=0.0)
    size(res_T) == size(Θ) || throw(DimensionMismatch("res_T must match Θ size"))

    dΘ_dr = op.Dr * Θ
    Nr = size(Θ, 1)
    inner_idx, outer_idx = _boundary_indices(op, Nr)
    apply_thermal_bc_at!(res_T, Θ, dΘ_dr, inner, value_inner, flux_inner, inner_idx)
    apply_thermal_bc_at!(res_T, Θ, dΘ_dr, outer, value_outer, flux_outer, outer_idx)
    return nothing
end

function apply_thermal_bc_at!(res_T, Θ, dΘ_dr,
                              bc::Symbol,
                              value::Real,
                              flux::Real,
                              idx::Int)
    if bc === :fixed_temperature
        res_T[idx, :] .= Θ[idx, :] .- value
    elseif bc === :fixed_flux
        res_T[idx, :] .= dΘ_dr[idx, :] .- flux
    else
        throw(ArgumentError("Unsupported thermal boundary condition: $(bc)"))
    end
end

function _boundary_indices(op, Nr::Int)
    if hasproperty(op, :r)
        r = op.r
        length(r) == Nr || throw(DimensionMismatch("r must have length $Nr"))
        return r[1] < r[end] ? (1, Nr) : (Nr, 1)
    end
    if hasproperty(op, :inv_r)
        inv_r = _get_inv_r(op, Nr)
        inv_r_vec = _inv_r_vector(inv_r, Nr)
        return inv_r_vec[1] > inv_r_vec[end] ? (1, Nr) : (Nr, 1)
    end
    return Nr, 1
end

function _get_im_m(op)
    if hasproperty(op, :im_m)
        return op.im_m
    elseif hasproperty(op, :m)
        return im * op.m
    elseif hasproperty(op, :params) && hasproperty(op.params, :m)
        return im * op.params.m
    end
    throw(ArgumentError("op must define `m` or `im_m` for azimuthal wavenumber"))
end

function _get_inv_r(op, Nr::Int)
    if hasproperty(op, :inv_r)
        inv_r = op.inv_r
        size(inv_r, 1) == Nr || throw(DimensionMismatch("inv_r must have $Nr rows"))
        return inv_r
    elseif hasproperty(op, :r)
        r = op.r
        length(r) == Nr || throw(DimensionMismatch("r must have length $Nr"))
        return 1.0 ./ r
    end
    throw(ArgumentError("op must define `inv_r` or `r`"))
end

function _get_inv_r_sinθ(op, inv_r, Nr::Int, Nθ::Int)
    if hasproperty(op, :inv_r_sinθ)
        inv_r_sinθ = op.inv_r_sinθ
        size(inv_r_sinθ, 1) == Nr || throw(DimensionMismatch("inv_r_sinθ must have $Nr rows"))
        size(inv_r_sinθ, 2) == Nθ || throw(DimensionMismatch("inv_r_sinθ must have $Nθ columns"))
        return inv_r_sinθ
    end

    sinθ = _get_sinθ(op, Nθ)
    inv_sinθ = 1.0 ./ sinθ
    inv_r_vec = _inv_r_vector(inv_r, Nr)
    return inv_r_vec .* inv_sinθ'
end

function _get_sinθ(op, Nθ::Int)
    if hasproperty(op, :sintheta)
        sinθ = op.sintheta
        length(sinθ) == Nθ || throw(DimensionMismatch("sintheta must have length $Nθ"))
        return sinθ
    elseif hasproperty(op, :sinθ)
        sinθ = getproperty(op, :sinθ)
        length(sinθ) == Nθ || throw(DimensionMismatch("sinθ must have length $Nθ"))
        return sinθ
    elseif hasproperty(op, :theta)
        θ = op.theta
        length(θ) == Nθ || throw(DimensionMismatch("theta must have length $Nθ"))
        return sin.(θ)
    elseif hasproperty(op, :θ)
        θ = getproperty(op, :θ)
        length(θ) == Nθ || throw(DimensionMismatch("θ must have length $Nθ"))
        return sin.(θ)
    end
    throw(ArgumentError("op must define `sintheta`, `sinθ`, `theta`, or `θ`"))
end

function _inv_r_vector(inv_r, Nr::Int)
    if ndims(inv_r) == 1
        length(inv_r) == Nr || throw(DimensionMismatch("inv_r must have length $Nr"))
        return inv_r
    elseif ndims(inv_r) == 2
        size(inv_r, 1) == Nr || throw(DimensionMismatch("inv_r must have $Nr rows"))
        return view(inv_r, :, 1)
    end
    throw(ArgumentError("inv_r must be a vector or matrix"))
end

function _inv_r_at(inv_r, idx::Int)
    if ndims(inv_r) == 1
        return inv_r[idx]
    elseif ndims(inv_r) == 2
        return inv_r[idx, 1]
    end
    throw(ArgumentError("inv_r must be a vector or matrix"))
end

# =============================================================================
#  Magnetic Field Boundary Conditions
# =============================================================================

"""
    spherical_bessel_j_logderiv(l::Int, x::Complex{T}) -> Complex{T}

Compute the logarithmic derivative of the spherical Bessel function of the first kind:

```math
\\frac{d}{dx}[\\log(j_l(x))] = \\frac{j'_l(x)}{j_l(x)}
```

# Mathematical Background

The spherical Bessel function jₗ(x) satisfies the recurrence relation:
```math
j'_l(x) = \\frac{l}{x} j_l(x) - j_{l+1}(x)
```

Therefore, the logarithmic derivative is:
```math
\\frac{j'_l(x)}{j_l(x)} = \\frac{l}{x} - \\frac{j_{l+1}(x)}{j_l(x)}
```

# Application: Conducting Inner Core

For a conducting inner core with finite conductivity σ, the magnetic boundary
condition couples the field and its derivative through a complex Bessel wavenumber:

```math
k = (1-i)\\sqrt{\\frac{\\omega}{2E_m}}
```

where ω is the oscillation frequency and Eₘ is the magnetic Ekman number.

The boundary condition becomes:
```math
f(r_i) - k \\cdot \\frac{j'_l(kr_i)}{j_l(kr_i)} \\cdot f'(r_i) = 0
```

# Numerical Stability

- For |x| < 10⁻¹⁰, uses series expansion: j'ₗ/jₗ ≈ l/x
- For normal values, uses recurrence relation
- Handles complex arguments robustly (needed for k = (1-i)√...)

# Arguments

- `l::Int`: Spherical harmonic degree (l ≥ 0)
- `x::Complex{T}`: Complex argument (typically k·r)

# Returns

- `Complex{T}`: The logarithmic derivative j'ₗ(x)/jₗ(x)

# Examples

```julia
using SpecialFunctions

# Real argument
l = 2
x = 1.5
logderiv = spherical_bessel_j_logderiv(l, x)

# Complex argument (conducting boundary)
Em = 1e-3
omega = 0.1 + 0.5im  # Complex frequency
k = (1 - 1im) * sqrt(omega / (2*Em))
ri = 0.35
logderiv_complex = spherical_bessel_j_logderiv(l, k * ri)
```

# References

- Kore implementation: kore-main/bin/utils.py, lines 487-526
- Satapathy (2013): Boundary conditions for conducting cores
- Zhang & Fearn (1994): Hydromagnetic flow in planetary cores

# See Also

- [`apply_magnetic_boundary_conditions!`](@ref): Uses this for bci_magnetic=1
- `SpecialFunctions.sphericalbesselj`: Underlying Bessel function
"""
function spherical_bessel_j_logderiv(l::Int, x::Complex{T}) where {T<:Real}
    # For very small |x|, use series expansion: j_l(x) ≈ x^l / (2l+1)!!
    # so d/dx[log(j_l)] ≈ l/x
    if abs(x) < 1e-10
        return complex(T(l)) / x
    end

    # Use recurrence relation: d/dx[log(j_l)] = l/x - j_{l+1}/j_l
    jl = sphericalbesselj(l, x)
    jl_plus_1 = sphericalbesselj(l + 1, x)

    # Check for numerical issues
    if abs(jl) < 1e-30
        # If j_l is very small, fall back to asymptotic form
        return complex(T(l)) / x
    end

    return T(l) / x - jl_plus_1 / jl
end

# Overload for real arguments (though we primarily use complex)
spherical_bessel_j_logderiv(l::Int, x::T) where {T<:Real} =
    spherical_bessel_j_logderiv(l, complex(x))

"""
    apply_magnetic_boundary_conditions!(A, B, op, section)

Apply boundary conditions to magnetic field sections using the tau method.

Modifies the sparse matrices A and B in-place by replacing boundary rows with
constraint equations. This implements the tau method for enforcing boundary
conditions in spectral methods.

# Physical Boundary Conditions

## Boundary Condition Types (bci_magnetic, bco_magnetic)

- **0 = Insulating**: Electrically insulating boundary (mantle, vacuum)
  - No currents can flow across the boundary
  - Magnetic field lines can penetrate but with specific matching conditions
  - Most common for CMB (electrically insulating silicate mantle)

- **1 = Conducting**: Finite electrical conductivity (partially implemented)
  - Allows currents in a thin layer with finite conductivity σ
  - Uses Bessel functions with complex wavenumber k = (1-i)√(ω/2Eₘ)

- **2 = Perfect Conductor**: Infinite electrical conductivity
  - Tangential electric field must vanish: E_tangential = 0
  - Used for highly conducting solid inner cores (e.g., Earth's iron core)
  - Requires 2-row BC for poloidal field, 1-row for toroidal

# Mathematical Formulation

## Poloidal Magnetic Field (section = :f)

The poloidal magnetic field is represented by scalar f(r) such that:
```math
\\mathbf{B}_{pol} = \\nabla \\times (f(r) \\mathbf{r})
```

### Insulating Boundaries

Match normal component and tangential derivative:

**CMB (r = rₒ = 1):**
```math
(l+1) f(r_o) + r_o f'(r_o) = 0
```

**ICB (r = rᵢ = ricb):**
```math
l f(r_i) - r_i f'(r_i) = 0
```

*Reference*: Kore assemble.py:1494-1572; Hollerbach (2000), Phys. Earth Planet. Inter.

### Conducting ICB (Finite Conductivity)

With skin depth δ ~ √(η/ω), the boundary condition involves Bessel functions:
```math
f(r_i) - k \\cdot \\frac{j'_l(kr_i)}{j_l(kr_i)} \\cdot f'(r_i) = 0
```
where k = (1-i)√(ω/2Eₘ) is the complex magnetic diffusion wavenumber.

*Reference*: Kore assemble.py:1575-1612; Satapathy (2013)

### Perfect Conductor ICB

Requires **two boundary conditions** (2-row constraint):

**Row 1**: No radial field penetration:
```math
f(r_i) = 0
```

**Row 2**: Tangential electric field vanishes:
```math
E_m \\left( -f''(r_i) - \\frac{2}{r_i}f'(r_i) + \\frac{L}{r_i^2}f(r_i) \\right) = 0
```

where L = l(l+1).

*Reference*: Kore assemble.py:1614-1630; Dormy et al. (2004), J. Fluid Mech.

### Conducting/Perfect CMB

Simply enforce no penetration:
```math
f(r_o) = 0
```

## Toroidal Magnetic Field (section = :g)

The toroidal magnetic field is:
```math
\\mathbf{B}_{tor} = \\nabla \\times \\nabla \\times (g(r) \\mathbf{r})
```

### Insulating/Conducting (Standard Cases)

No toroidal field can exist outside an insulating or finitely conducting boundary:
```math
g(r) = 0 \\quad \\text{at both boundaries}
```

### Perfect Conductor ICB

Tangential electric field condition gives:
```math
E_m \\left( -g'(r_i) - \\frac{1}{r_i}g(r_i) \\right) = 0
```

*Reference*: Kore assemble.py:1631-1641

# Implementation: Tau Method

The tau method replaces matrix rows with boundary condition equations:

1. **Zero out row**: Set A[row,:] = 0 and B[row,:] = 0
2. **Build constraint**: Evaluate boundary condition using Chebyshev basis
   - Function values: Tₙ(1) = 1 at CMB, Tₙ(-1) = (-1)ⁿ at ICB
   - Derivatives: Evaluate Chebyshev derivatives at the boundary points
3. **Set row**: A[row,:] = boundary constraint, B[row,:] = 0

# Arguments

- `A::SparseMatrixCSC`: Left-hand side matrix (modified in-place)
- `B::SparseMatrixCSC`: Right-hand side matrix (modified in-place)
- `op::MHDStabilityOperator`: Operator containing parameters and mode structure
- `section::Symbol`: Which magnetic field component (:f for poloidal, :g for toroidal)

# Matrix Structure

For each l-mode, modifies rows corresponding to boundary points:
- First row in each mode block: Outer boundary (CMB)
- Last row in each mode block: Inner boundary (ICB)
- Perfect conductor: Uses TWO rows at ICB for section :f

# Examples

```julia
# Create operator
params = MHDParams(E=1e-3, Pm=5.0, ricb=0.35, m=2, lmax=15, N=32,
                   bci_magnetic=0, bco_magnetic=0)  # Insulating
op = MHDStabilityOperator(params)

# Matrices before BC
A, B = assemble_mhd_matrices_no_bc(op)  # Hypothetical function

# Apply magnetic BCs
apply_magnetic_boundary_conditions!(A, B, op, :f)  # Poloidal field
apply_magnetic_boundary_conditions!(A, B, op, :g)  # Toroidal field

# Now A, B have correct boundary conditions enforced
```

# Boundary Condition Compatibility

| bci_magnetic | bco_magnetic | Physical Scenario | Status |
|--------------|--------------|-------------------|--------|
| 0 | 0 | Insulating mantle & vacuum outside core | ✅ Implemented |
| 0 | 1,2 | Insulating ICB, conducting CMB | ✅ Implemented (rare) |
| 2 | 0 | Perfect conductor ICB, insulating CMB | ✅ **NEW!** Earth-like |
| 1 | 0 | Conducting ICB (finite σ) | ⚠️ Infrastructure ready |

# References

- Hollerbach (2000), "A spectral solution of the magneto-convection equations...",
  Physics of the Earth and Planetary Interiors 117, 319-333
- Dormy et al. (2004), "MHD flow in a slightly differentially rotating spherical shell",
  Journal of Fluid Mechanics 501, 43-70
- Satapathy (2013), "Dynamo action in a rotating spherical shell with an outer stably stratified layer"
- Kore implementation: kore-main/bin/assemble.py:1472-1786

# See Also

- [`spherical_bessel_j_logderiv`](@ref): For conducting boundaries
- [`MHDParams`](@ref): Parameter structure with magnetic BC options
- [`assemble_mhd_matrices`](@ref): Calls this function during assembly
"""
function apply_magnetic_boundary_conditions!(A::SparseMatrixCSC,
                                            B::SparseMatrixCSC,
                                            op,
                                            section::Symbol)
    params = op.params
    N = params.N
    n_per_mode = N + 1
    ri = params.ricb
    ro = one(typeof(ri))  # Outer radius normalized to 1

    nb_u = length(op.ll_u)
    nb_v = length(op.ll_v)
    nb_f = length(op.ll_f)
    nb_g = length(op.ll_g)
    scale = UltrasphericalSpectral._radial_scale(ri, ro)
    r_outer = UltrasphericalSpectral._boundary_radius(ri, ro, :outer)
    r_inner = UltrasphericalSpectral._boundary_radius(ri, ro, :inner)
    outer_vals = UltrasphericalSpectral._chebyshev_boundary_values(N, :outer)
    inner_vals = UltrasphericalSpectral._chebyshev_boundary_values(N, :inner)
    outer_deriv = scale * UltrasphericalSpectral._chebyshev_boundary_derivative(N, :outer)
    inner_deriv = scale * UltrasphericalSpectral._chebyshev_boundary_derivative(N, :inner)
    inner_second = scale^2 * UltrasphericalSpectral._chebyshev_boundary_second_derivative(N, :inner)

    if section == :f  # Poloidal magnetic field
        for (k, l) in enumerate(op.ll_f)
            # Offset to f section: after u and v sections
            row_base = (nb_u + nb_v + k - 1) * n_per_mode
            block_range = (row_base + 1):(row_base + n_per_mode)

            # ----------------------------------------------------------------
            # Outer boundary (CMB, r = ro = 1)
            # ----------------------------------------------------------------
            row_cmb = row_base + 1

            if params.bco_magnetic == 0
                # Insulating CMB: (l+1)·f(ro) + ro·f'(ro) = 0
                # Following Kore: kore-main/bin/assemble.py:1494-1509

                # Zero out row
                A[row_cmb, :] .= zero(ComplexF64)
                B[row_cmb, :] .= zero(ComplexF64)

                # Build constraint: (l+1)·f + ro·f'
                A[row_cmb, block_range] = ComplexF64.((l + 1) * outer_vals + r_outer * outer_deriv)

            else
                # Perfectly conducting: f = 0 (no penetration)
                A[row_cmb, :] .= zero(ComplexF64)
                B[row_cmb, :] .= zero(ComplexF64)
                A[row_cmb, block_range] = ComplexF64.(outer_vals)
            end

            # ----------------------------------------------------------------
            # Inner boundary (ICB, r = ri)
            # ----------------------------------------------------------------
            row_icb = row_base + n_per_mode

            if params.bci_magnetic == 0
                # Insulating ICB: l·f(ri) - ri·f'(ri) = 0
                # Following Kore: kore-main/bin/assemble.py:1548-1572

                # Zero out row
                A[row_icb, :] .= zero(ComplexF64)
                B[row_icb, :] .= zero(ComplexF64)

                # Build constraint: l·f - ri·f'
                A[row_icb, block_range] = ComplexF64.(l * inner_vals - r_inner * inner_deriv)

            elseif params.bci_magnetic == 1
                freq = params.forcing_frequency
                Em = params.Em
                if Em <= 0
                    error("Conducting magnetic BC requires Em > 0")
                end

                # If frequency is zero (steady state), condition reduces to f(ri) = 0
                if iszero(freq)
                    A[row_icb, :] .= zero(ComplexF64)
                    B[row_icb, :] .= zero(ComplexF64)
                    A[row_icb, block_range] = ComplexF64.(r_inner .* inner_vals .- l .* inner_deriv)
                else
                    k_wave = (1 - 1im) * sqrt(complex(freq) / (2 * Em))
                    dlog = spherical_bessel_j_logderiv(l, k_wave * ri)

                    A[row_icb, :] .= zero(ComplexF64)
                    B[row_icb, :] .= zero(ComplexF64)
                    A[row_icb, block_range] = ComplexF64.(inner_vals) .-
                                             k_wave * dlog .* ComplexF64.(inner_deriv)
                end

            elseif params.bci_magnetic == 2
                # Perfect conductor ICB: 2-row boundary condition
                # Row 1: f = 0
                # Row 2: Em·(-f'' - 2/ri·f' + L/ri²·f) = 0
                # Following Kore: kore-main/bin/assemble.py:1614-1630

                L = l * (l + 1)

                # Row 1: f(ri) = 0
                A[row_icb, :] .= zero(ComplexF64)
                B[row_icb, :] .= zero(ComplexF64)
                A[row_icb, block_range] = ComplexF64.(inner_vals)

                # Row 2: Em·(-f'' - (2/ri)·f' + (L/ri²)·f) = 0
                # We need to use the row BEFORE row_icb (row_icb-1) for the second BC
                # This replaces the last interior point
                row_icb2 = row_icb - 1

                # Zero out row
                A[row_icb2, :] .= zero(ComplexF64)
                B[row_icb2, :] .= zero(ComplexF64)
                value_term = (L / ri^2) .* inner_vals
                deriv1_term = -(2.0 / ri) .* inner_deriv
                deriv2_term = -inner_second
                A[row_icb2, block_range] = ComplexF64.(params.Em * (value_term + deriv1_term + deriv2_term))

            else
                # Simple conducting: f = 0 (no penetration)
                A[row_icb, :] .= zero(ComplexF64)
                B[row_icb, :] .= zero(ComplexF64)
                A[row_icb, block_range] = ComplexF64.(inner_vals)
            end
        end

    elseif section == :g  # Toroidal magnetic field
        for (k, l) in enumerate(op.ll_g)
            # Offset to g section: after u, v, f sections
            row_base = (nb_u + nb_v + nb_f + k - 1) * n_per_mode
            block_range = (row_base + 1):(row_base + n_per_mode)

            # ----------------------------------------------------------------
            # Outer boundary (CMB): g = 0 (for all BC types)
            # ----------------------------------------------------------------
            # Following Kore: kore-main/bin/assemble.py:1511-1522
            row_cmb = row_base + 1
            A[row_cmb, :] .= zero(ComplexF64)
            B[row_cmb, :] .= zero(ComplexF64)
            A[row_cmb, block_range] = ComplexF64.(outer_vals)

            # ----------------------------------------------------------------
            # Inner boundary (ICB)
            # ----------------------------------------------------------------
            row_icb = row_base + n_per_mode

            if params.bci_magnetic == 0
                # Insulating: g = 0
                A[row_icb, :] .= zero(ComplexF64)
                B[row_icb, :] .= zero(ComplexF64)
                A[row_icb, block_range] = ComplexF64.(inner_vals)

            elseif params.bci_magnetic == 1
                freq = params.forcing_frequency
                Em = params.Em
                if Em <= 0
                    error("Conducting magnetic BC requires Em > 0")
                end

                if iszero(freq)
                    A[row_icb, :] .= zero(ComplexF64)
                    B[row_icb, :] .= zero(ComplexF64)
                    A[row_icb, block_range] = ComplexF64.(r_inner .* inner_vals .- l .* inner_deriv)
                else
                    k_wave = (1 - 1im) * sqrt(complex(freq) / (2 * Em))
                    dlog = spherical_bessel_j_logderiv(l, k_wave * ri)

                    A[row_icb, :] .= zero(ComplexF64)
                    B[row_icb, :] .= zero(ComplexF64)
                    A[row_icb, block_range] = ComplexF64.(inner_vals) .-
                                             k_wave * dlog .* ComplexF64.(inner_deriv)
                end

            elseif params.bci_magnetic == 2
                # Perfect conductor: Em·(-g' - 1/ri·g) = 0
                # Following Kore: kore-main/bin/assemble.py:1631-1641

                # Zero out row
                A[row_icb, :] .= zero(ComplexF64)
                B[row_icb, :] .= zero(ComplexF64)
                value_term = -(1.0 / ri) .* inner_vals
                deriv1_term = -inner_deriv
                A[row_icb, block_range] = ComplexF64.(params.Em * (value_term + deriv1_term))

            else
                # Default: g = 0
                A[row_icb, :] .= zero(ComplexF64)
                B[row_icb, :] .= zero(ComplexF64)
                A[row_icb, block_range] = ComplexF64.(inner_vals)
            end
        end
    end

    return nothing
end
