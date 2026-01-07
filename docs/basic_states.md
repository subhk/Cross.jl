# Basic States

Cross.jl separates the base (steady) state of the system from the perturbations whose stability we study. This mirrors Kore’s layered approach: build the reference configuration, then add physics. Two data structures handle the base state:

- `BasicState` – axisymmetric (m = 0) backgrounds for classical onset problems.
- `BasicState3D` – fully 3-D backgrounds with longitudinal variation for tri-global analysis.

## Axisymmetric States (`BasicState`)

Axisymmetric cases keep only spherical harmonic modes with azimuthal index m = 0. The helper `conduction_basic_state` builds the default conduction profile that sets the inner boundary temperature to unity and the outer boundary to zero:

```julia
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
bs = conduction_basic_state(cd, χ, lmax_bs = 6)
```

Fields inside `BasicState`:

- `theta_coeffs[ℓ]` – temperature coefficients.
- `uphi_coeffs[ℓ]` – zonal velocity component from thermal wind balance.
- `dtheta_dr_coeffs[ℓ]`, `duphi_dr_coeffs[ℓ]` – radial derivatives used by the operator.

To include the base state in an onset problem:

```julia
params = ShellParams(
    E = 1e-5,
    Pr = 1.0,
    Ra = 1e7,
    χ = 0.35,
    m = 12,
    lmax = 60,
    Nr = 96,
    basic_state = bs,
)
```

Cross.jl automatically augments the operator with advection terms derived from the supplied coefficients.

### Meridional Variations

The prototype `meridional_basic_state` adds a controlled `Y_{20}` temperature perturbation on top of conduction:

```julia
bs_meridional = meridional_basic_state(cd, χ, Ra, Pr, lmax_bs = 6, amplitude = 0.05)
```

This function illustrates how to generate higher-degree `θ̄_ℓ0` and `ū_φ,ℓ0` components. It is still under active development; inspect the TODOs in `src/linear_stability.jl` before relying on it for production runs.

## Fully 3-D States (`BasicState3D`)

`BasicState3D` stores coefficients indexed by `(ℓ, m)` pairs and includes all three velocity components (`u_r`, `u_θ`, `u_φ`). Use it when boundary forcing or flow patterns vary with longitude.

```julia
bs3d = BasicState3D(
    lmax_bs = 8,
    mmax_bs = 3,
    Nr = Nr,
    r = cd.x,
    theta_coeffs = Dict((ℓ, m) => zeros(Nr) for ℓ in 0:8, m in -3:3),
    dtheta_dr_coeffs = Dict((ℓ, m) => zeros(Nr) for ℓ in 0:8, m in -3:3),
    ur_coeffs = Dict((ℓ, m) => zeros(Nr) for ℓ in 0:8, m in -3:3),
    utheta_coeffs = Dict((ℓ, m) => zeros(Nr) for ℓ in 0:8, m in -3:3),
    uphi_coeffs = Dict((ℓ, m) => zeros(Nr) for ℓ in 0:8, m in -3:3),
    dur_dr_coeffs = Dict((ℓ, m) => zeros(Nr) for ℓ in 0:8, m in -3:3),
    dutheta_dr_coeffs = Dict((ℓ, m) => zeros(Nr) for ℓ in 0:8, m in -3:3),
    duphi_dr_coeffs = Dict((ℓ, m) => zeros(Nr) for ℓ in 0:8, m in -3:3),
)
```

Populate the dictionaries with coefficients exported from other solvers (for example, thermal wind solutions or MHD simulations). Cross.jl does not prescribe how to generate them; it only consumes the spectral amplitudes.

## Saving and Loading

Since base states can take time to build, save them with JLD2:

```julia
using JLD2
@save "basic_states/conduction_l6.jld2" bs
@load "basic_states/conduction_l6.jld2" bs_loaded
```

The loaded object behaves like the original and can be plugged into `ShellParams`.

## Checklist

- [ ] `BasicState`/`BasicState3D` radial grids match the analysis grid (`Nr`, `χ`).
- [ ] Coefficients satisfy the required symmetries (e.g., reality conditions).
- [ ] Dictionaries contain entries for all `(ℓ, m)` pairs you expect.
- [ ] Saved JLD2 files reload without conversion warnings.
