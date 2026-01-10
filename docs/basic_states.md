# Basic States

Cross.jl separates the base (steady) state of the system from the perturbations whose stability we study. This enables analyzing convection onset with realistic background temperature and flow profiles.

## Overview

Two data structures handle base states:

| Type | Use Case | Description |
|------|----------|-------------|
| `BasicState` | Axisymmetric ($m=0$) | Classical onset problems with zonally-symmetric backgrounds |
| `BasicState3D` | Non-axisymmetric | Tri-global analysis with longitudinal variations |

## Axisymmetric States (`BasicState`)

Axisymmetric cases keep only spherical harmonic modes with azimuthal index $m = 0$.

### Structure

```julia
struct BasicState{T}
    lmax_bs::Int
    Nr::Int
    r::Vector{T}
    theta_coeffs::Dict{Int, Vector{T}}
    uphi_coeffs::Dict{Int, Vector{T}}
    dtheta_dr_coeffs::Dict{Int, Vector{T}}
    duphi_dr_coeffs::Dict{Int, Vector{T}}
end
```

### Conduction Basic State

The simplest case is pure conduction with no flow:

```julia
using Cross

# Create Chebyshev differentiation matrices
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# Build conduction state
bs = conduction_basic_state(cd, χ, lmax_bs=6)
```

The conduction profile satisfies $\nabla^2 \bar{T} = 0$ with:
- $\bar{T}(r_i) = 1$ (hot inner boundary)
- $\bar{T}(r_o) = 0$ (cold outer boundary)

Solution:
$$
\bar{T}(r) = \frac{r_o/r - 1}{r_o/r_i - 1}
$$

### Meridional Variations

Add a $Y_{2,0}$ temperature perturbation for pole-equator differential heating:

```julia
bs_meridional = meridional_basic_state(
    cd,          # Chebyshev differentiation
    χ,           # Radius ratio
    E,           # Ekman number
    Ra,          # Rayleigh number
    Pr,          # Prandtl number
    lmax_bs = 6,
    amplitude = 0.05;
    mechanical_bc = :no_slip,
)
```

This generates:
- Temperature perturbation: $\bar{\Theta}_{20}(r) \propto Y_{2,0}(\theta)$
- Thermal wind: $\bar{u}_\phi(r,\theta)$ from thermal wind balance

### Thermal Wind Balance

When temperature varies with latitude, geostrophic balance requires a zonal flow:

$$
2\Omega \cos\theta \frac{\partial \bar{u}_\phi}{\partial r} = -\frac{Ra \cdot E^2}{Pr \cdot r} \frac{\partial \bar{\Theta}}{\partial \theta}
$$

This balance is handled internally by `meridional_basic_state`. For custom
axisymmetric profiles, you can call the solver directly:

```julia
solve_thermal_wind_balance!(
    uphi_coeffs,
    duphi_dr_coeffs,
    theta_coeffs,
    cd, χ, 1.0, Ra, Pr;
    mechanical_bc = :no_slip,
    E = E,
)
```

### Using Basic States in Problems

Pass the basic state to `ShellParams`:

```julia
params = ShellParams(
    E = 1e-5,
    Pr = 1.0,
    Ra = 1e7,
    χ = 0.35,
    m = 12,
    lmax = 60,
    Nr = 96,
    basic_state = bs,  # Include the basic state
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)
```

Cross.jl automatically augments the linearized operator with advection terms:

$$
\mathbf{u}' \cdot \nabla \bar{\mathbf{u}} + \bar{\mathbf{u}} \cdot \nabla \mathbf{u}'
$$

## Fully 3-D States (`BasicState3D`)

`BasicState3D` stores coefficients indexed by $(\ell, m)$ pairs for non-axisymmetric backgrounds.

### Structure

```julia
struct BasicState3D{T}
    # Grid
    r::Vector{T}
    Nr::Int
    lmax_bs::Int
    mmax_bs::Int

    # Temperature: θ̄_ℓm(r)
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    dtheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}

    # Velocity: ū_r,ℓm(r), ū_θ,ℓm(r), ū_φ,ℓm(r)
    ur_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    utheta_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}}

    # Velocity derivatives
    dur_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    dutheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    duphi_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
end
```

### Creating 3-D Basic States

#### From Boundary Conditions

Use `nonaxisymmetric_basic_state` to solve $\nabla^2 \bar{T} = 0$ with specified boundary modes:

```julia
boundary_modes = Dict(
    (2, 0) => 0.1,    # Y₂₀ amplitude at boundary
    (2, 2) => 0.05,   # Y₂₂ amplitude at boundary
)

E = 1e-5

bs3d = nonaxisymmetric_basic_state(
    cd,               # Chebyshev differentiation
    χ,                # Radius ratio
    E,                # Ekman number
    Ra,               # Rayleigh number
    Pr,               # Prandtl number
    8,
    4,
    boundary_modes,
)
```

#### Manual Construction

For custom profiles imported from other sources:

```julia
# Initialize empty dictionaries
Nr = 64
lmax_bs = 8
mmax_bs = 3
r = cd.x

theta_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}()
dtheta_dr_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}()

# Populate for all (ℓ,m) pairs
for l in 0:lmax_bs
    for m in -min(l, mmax_bs):min(l, mmax_bs)
        theta_coeffs[(l, m)] = zeros(Nr)
        dtheta_dr_coeffs[(l, m)] = zeros(Nr)
    end
end

# Set specific mode amplitudes
theta_coeffs[(2, 0)] .= your_temperature_profile

# Create the BasicState3D
bs3d = BasicState3D(
    r = r,
    Nr = Nr,
    lmax_bs = lmax_bs,
    mmax_bs = mmax_bs,
    theta_coeffs = theta_coeffs,
    dtheta_dr_coeffs = dtheta_dr_coeffs,
    # ... velocity coefficients ...
)
```

### Importing from External Codes

To import coefficients from other simulation codes (e.g., Rayleigh, Magic):

1. **Export spectral coefficients** from the source code
2. **Transform to Cross.jl convention** (check normalization)
3. **Populate the dictionaries** with radially interpolated values
4. **Compute derivatives** using Chebyshev differentiation

```julia
# Example: importing from external data
using JLD2

# Load external data
@load "external_basic_state.jld2" theta_lm r_ext

# Interpolate to Cross.jl grid
using Interpolations
for (lm, coeffs) in theta_lm
    itp = LinearInterpolation(r_ext, coeffs)
    theta_coeffs[lm] = itp.(cd.x)
    dtheta_dr_coeffs[lm] = cd.D1 * theta_coeffs[lm]
end
```

## Mode Coupling with Basic States

When a non-axisymmetric basic state is present, perturbation modes couple through advection:

$$
Y_{\ell_1, m_1} \times Y_{\ell_2, m_2} = \sum_{\ell'} G_{\ell_1 \ell_2 \ell'}^{m_1 m_2 m'} Y_{\ell', m_1+m_2}
$$

Where $G$ is the Gaunt coefficient computed from Wigner 3j symbols:

$$
G_{\ell_1 \ell_2 \ell_3}^{m_1 m_2 m_3} = \sqrt{\frac{(2\ell_1+1)(2\ell_2+1)(2\ell_3+1)}{4\pi}}
\begin{pmatrix} \ell_1 & \ell_2 & \ell_3 \\ 0 & 0 & 0 \end{pmatrix}
\begin{pmatrix} \ell_1 & \ell_2 & \ell_3 \\ m_1 & m_2 & m_3 \end{pmatrix}
$$

This coupling is handled automatically by `BasicStateOperators`:

```julia
bs_ops = build_basic_state_operators(bs3d, params)
add_basic_state_operators!(A, B, bs_ops, block_indices)
```

## Saving and Loading

Since base states can be expensive to compute, save them with JLD2:

```julia
using JLD2

# Save
@save "basic_states/meridional_l6.jld2" bs

# Load
@load "basic_states/meridional_l6.jld2" bs_loaded

# Use in new problem
params = ShellParams(..., basic_state = bs_loaded)
```

## Reality Conditions

For real physical fields, spectral coefficients must satisfy:

$$
\bar{f}_{\ell,-m} = (-1)^m \bar{f}_{\ell,m}^*
$$

When constructing `BasicState3D` manually, ensure this condition holds:

```julia
for l in 0:lmax_bs
    for m in 1:min(l, mmax_bs)
        theta_coeffs[(l, -m)] = (-1)^m * conj(theta_coeffs[(l, m)])
    end
end
```

## Examples

### Example 1: Conduction with Thermal Wind

```julia
using Cross

# Setup
E = 1e-5
Pr = 1.0
Ra = 1e7
χ = 0.35
Nr = 64

cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# Create meridional basic state
bs = meridional_basic_state(cd, χ, E, Ra, Pr;
    lmax_bs = 6,
    amplitude = 0.1,
    mechanical_bc = :no_slip,
)

# Verify structure
println("Temperature modes: ", keys(bs.theta_coeffs))
println("Zonal flow modes: ", keys(bs.uphi_coeffs))
```

### Example 2: Non-Axisymmetric Boundary Forcing

```julia
# Hemispheric heating pattern
boundary_modes = Dict(
    (1, 0) => 0.0,      # No ℓ=1, m=0 (no CMB heat flux variation)
    (2, 0) => 0.15,     # Equator-pole variation
    (2, 2) => 0.08,     # East-west variation
    (3, 2) => 0.03,     # Higher-order structure
)

bs3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 10, 4, boundary_modes)

# Use with tri-global analysis
tri_params = TriglobalParams(
    E = E, Pr = Pr, Ra = Ra, χ = χ,
    m_range = -3:3,
    lmax = 40,
    Nr = Nr,
    basic_state_3d = bs3d,
)
```

## Checklist

Before using a basic state:

- [ ] Radial grids match between basic state and analysis (`Nr`, `χ`)
- [ ] Coefficients satisfy reality conditions for physical fields
- [ ] All expected $(\ell, m)$ pairs have entries in dictionaries
- [ ] Derivatives computed consistently with Chebyshev operators
- [ ] Saved JLD2 files reload without conversion warnings

## Next Steps

- **[Tri-Global Analysis](triglobal.md)** - Use 3-D basic states for mode coupling
- **[MHD Extension](mhd_extension.md)** - Add magnetic field effects to basic states

---

!!! info "Example Scripts"
    See `example/basic_state_onset_example.jl` and `example/nonaxisymmetric_basic_state.jl` for working examples.
