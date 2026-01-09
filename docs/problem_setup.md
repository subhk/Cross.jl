# Setting Up Your First Problem

This tutorial walks you through assembling a linear stability operator for a rotating spherical shell, finding the critical Rayleigh number, and analyzing the results.

## Overview

By the end of this guide you will:

1. Assemble a linear stability operator for a rotating spherical shell
2. Search for the critical Rayleigh number
3. Inspect the leading eigenmode structure
4. Persist results for reuse

## Step 1: Define Physical and Numerical Parameters

The `ShellParams` helper converts your inputs into a consistent `OnsetParams` struct used internally.

```julia
using Cross

params = ShellParams(
    # Physical parameters
    E = 3e-6,              # Ekman number
    Pr = 1.0,              # Prandtl number
    Ra = 5e6,              # Initial Rayleigh guess

    # Geometry
    χ = 0.35,              # Radius ratio r_i / r_o
    ri = 0.35,             # Inner radius (optional, derived from χ)
    ro = 1.0,              # Outer radius

    # Spectral resolution
    m = 8,                 # Azimuthal wavenumber
    lmax = 80,             # Maximum spherical harmonic degree
    Nr = 96,               # Radial grid resolution

    # Boundary conditions
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)
```

### Parameter Reference

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `E` | Float64 | Ekman number $\nu/(\Omega L^2)$ | $E > 0$ |
| `Pr` | Float64 | Prandtl number $\nu/\kappa$ | $Pr > 0$ |
| `Ra` | Float64 | Rayleigh number | $Ra \geq 0$ |
| `χ` | Float64 | Radius ratio $r_i/r_o$ | $0 < \chi < 1$ |
| `m` | Int | Azimuthal wavenumber | $m \geq 0$ |
| `lmax` | Int | Maximum spherical harmonic degree | $lmax \geq m$ |
| `Nr` | Int | Radial resolution | $Nr \geq 4$ |
| `mechanical_bc` | Symbol | Velocity boundary condition | `:no_slip` or `:stress_free` |
| `thermal_bc` | Symbol | Temperature boundary condition | `:fixed_temperature` or `:fixed_flux` |
| `use_sparse_weighting` | Bool | Use sparse tau weighting | `true` or `false` |
| `equatorial_symmetry` | Symbol | Equatorial parity filter | `:both`, `:symmetric`, or `:antisymmetric` |

### Boundary Conditions

#### Mechanical Boundary Conditions

| Type | Symbol | Physical Meaning | Mathematical Form |
|------|--------|------------------|-------------------|
| No-slip | `:no_slip` | Fluid sticks to boundary | $\mathbf{u} = 0$ |
| Stress-free | `:stress_free` | Zero tangential stress | $P=0$, $r \partial_r^2 P = 0$, $-r \partial_r T + T = 0$ |

#### Thermal Boundary Conditions

| Type | Symbol | Physical Meaning | Mathematical Form |
|------|--------|------------------|-------------------|
| Fixed temperature | `:fixed_temperature` | Isothermal boundary | $\Theta = 0$ |
| Fixed flux | `:fixed_flux` | Insulating boundary | $\partial\Theta/\partial r = 0$ |

## Step 2: Inspect the Operator Structure

Build the linear stability operator and examine its properties:

```julia
op = LinearStabilityOperator(params)

println("Total degrees of freedom: ", op.total_dof)
println("ℓ-sets (poloidal): ", op.l_sets[:P])
println("ℓ-sets (toroidal): ", op.l_sets[:T])
println("ℓ-sets (temperature): ", op.l_sets[:Θ])
```

### Understanding Degrees of Freedom

The total DOF is determined by:

$$
\text{DOF} = N_r \times (N_\ell^{pol} + N_\ell^{tor} + N_\ell^{temp})
$$

Where:
- $N_r$ = radial points
- $N_\ell^{pol}$ = number of poloidal ℓ modes
- $N_\ell^{tor}$ = number of toroidal ℓ modes
- $N_\ell^{temp}$ = number of temperature ℓ modes

For a given azimuthal mode $m$, the allowed $\ell$ values satisfy:

- $\ell \geq m$ (spherical harmonic constraint)
- Parity selection (for equatorial symmetry filtering)

### Matrix Structure

The operator assembles sparse matrices $A$ and $B$ for the generalized eigenvalue problem:

$$
A \mathbf{x} = \sigma B \mathbf{x}
$$

Where $\mathbf{x} = [P_\ell(r), T_\ell(r), \Theta_\ell(r)]$ contains the spectral coefficients.

## Step 3: Find the Critical Rayleigh Number

Use `find_critical_rayleigh` to perform a bracket search:

```julia
Ra_c, ω_c, eigvec = find_critical_rayleigh(
    E = params.E,
    Pr = params.Pr,
    χ = params.χ,
    m = params.m,
    lmax = params.lmax,
    Nr = params.Nr;
    Ra_guess = params.Ra,
    mechanical_bc = params.mechanical_bc,
    thermal_bc = params.thermal_bc,
    nev = 6,
)

@info "Critical parameters found" Ra_c ω_c
```

### Search Algorithm

The critical Rayleigh number search:

1. Starts from `Ra_guess`
2. Evaluates growth rate $\sigma(Ra)$ at each iteration
3. Uses bisection to find $Ra$ where $\sigma = 0$
4. Converges when $|\sigma| < \text{tol}$

### Solver Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `nev` | Int | 6 | Number of eigenvalues to compute |
| `tol` | Float64 | 1e-8 | Convergence tolerance |
| `maxiter` | Int | 100 | Maximum iterations |
| `which` | Symbol | `:LR` | Eigenvalue selection (largest real part) |

## Step 4: Compute Growth Rates at Fixed Rayleigh

For parameter studies, use `find_growth_rate` to evaluate stability at fixed $Ra$:

```julia
# Single calculation
eigenvalues, eigenvectors, _, info = find_growth_rate(op; nev=8)

σ₁ = real(eigenvalues[1])
ω₁ = imag(eigenvalues[1])

if σ₁ > 0
    println("System is UNSTABLE (σ = $σ₁)")
else
    println("System is STABLE (σ = $σ₁)")
end
```

Or use `leading_modes` directly:

```julia
eigenvalues, eigenvectors, _, info = leading_modes(params;
    nev = 4,
    which = :LR,
    tol = 1e-6,
)
```

## Step 5: Reconstruct Physical Fields

Convert spectral eigenvectors back to physical space:

```julia
# Extract poloidal, toroidal, and temperature coefficients
eigvec = eigenvectors[1]

poloidal = Dict{Int, Vector{ComplexF64}}()
toroidal = Dict{Int, Vector{ComplexF64}}()
temperature = Dict{Int, Vector{ComplexF64}}()

for ℓ in op.l_sets[:P]
    poloidal[ℓ] = eigvec[op.index_map[(ℓ, :P)]]
end
for ℓ in op.l_sets[:T]
    toroidal[ℓ] = eigvec[op.index_map[(ℓ, :T)]]
end
for ℓ in op.l_sets[:Θ]
    temperature[ℓ] = eigvec[op.index_map[(ℓ, :Θ)]]
end

# If you already have P(r,θ) and T(r,θ) on a grid, you can compute velocities:
# u_r, u_θ, u_φ = potentials_to_velocity(P, T; Dr, Dθ, Lθ, r, sintheta, m)
```

### Field Reconstruction Functions

| Function | Output | Description |
|----------|--------|-------------|
| `potentials_to_velocity` | `(u_r, u_θ, u_φ)` | Velocity from grid-based poloidal/toroidal potentials |

## Step 6: Save and Load Results

Use JLD2 for efficient storage:

```julia
using JLD2

# Save results
@save "outputs/onset_case1.jld2" params Ra_c ω_c eigvec

# Load later
@load "outputs/onset_case1.jld2" params Ra_c ω_c eigvec
```

!!! tip "File Organization"
    Create an `outputs/` directory to organize your results by parameter set.

## Step 7: Parameter Sweeps

Automate scans over azimuthal modes or other parameters:

```julia
function sweep_azimuthal_modes(m_values; E, Pr, χ, lmax, Nr)
    results = Dict{Int, NamedTuple}()

    for m in m_values
        println("Computing m = $m...")
        try
            Ra_c, ω_c, _ = find_critical_rayleigh(
                E = E, Pr = Pr, χ = χ, m = m,
                lmax = max(lmax, m + 10),
                Nr = Nr,
                Ra_guess = 1e6,
            )
            results[m] = (Ra_c = Ra_c, ω_c = ω_c)
        catch err
            @warn "Failed for m=$m" err
            results[m] = (Ra_c = NaN, ω_c = NaN)
        end
    end

    return results
end

# Run sweep
results = sweep_azimuthal_modes(1:15;
    E = 1e-5, Pr = 1.0, χ = 0.35, lmax = 60, Nr = 64
)

# Find most unstable mode
m_crit = argmin(m -> results[m].Ra_c, keys(results))
println("Critical mode: m = $m_crit, Ra_c = $(results[m_crit].Ra_c)")
```

## Complete Example Script

Here's a complete workflow combining all steps:

```julia
#!/usr/bin/env julia
# complete_onset_analysis.jl

using Cross
using JLD2
using Printf

# === Configuration ===
E = 1e-5
Pr = 1.0
χ = 0.35
m_range = 5:15
lmax = 60
Nr = 64

# === Sweep over m ===
results = []

for m in m_range
    @printf("Processing m = %2d... ", m)

    params = ShellParams(
        E = E, Pr = Pr, Ra = 1e7, χ = χ,
        m = m, lmax = max(lmax, m + 10), Nr = Nr,
        mechanical_bc = :no_slip,
        thermal_bc = :fixed_temperature,
    )

    try
        Ra_c, ω_c, eigvec = find_critical_rayleigh(
            E = E, Pr = Pr, χ = χ, m = m,
            lmax = params.lmax, Nr = Nr,
            Ra_guess = 1e6,
        )
        push!(results, (m=m, Ra_c=Ra_c, ω_c=ω_c))
        @printf("Ra_c = %.4e, ω_c = %.4f\n", Ra_c, ω_c)
    catch err
        push!(results, (m=m, Ra_c=NaN, ω_c=NaN))
        @printf("FAILED\n")
    end
end

# === Find global minimum ===
valid = filter(r -> !isnan(r.Ra_c), results)
if !isempty(valid)
    critical = argmin(r -> r.Ra_c, valid)
    println("\n" * "="^50)
    @printf("Global critical point:\n")
    @printf("  m_c  = %d\n", critical.m)
    @printf("  Ra_c = %.6e\n", critical.Ra_c)
    @printf("  ω_c  = %.6f\n", critical.ω_c)
end

# === Save results ===
@save "outputs/onset_sweep.jld2" results E Pr χ
```

## Checklist

Before proceeding, verify:

- [ ] `ShellParams` constructed without assertion failures
- [ ] Operator degrees of freedom align with expectations
- [ ] `find_critical_rayleigh` converges from your `Ra_guess`
- [ ] Eigenvector converted back to physical fields without NaNs
- [ ] Results saved for later reuse

## Next Steps

- **[Basic States](basic_states.md)** - Construct custom background configurations
- **[Tri-Global Analysis](triglobal.md)** - Mode coupling for non-axisymmetric problems
- **[MHD Extension](mhd_extension.md)** - Add magnetic field effects

---

!!! info "Example Scripts"
    See `example/linear_stability_demo.jl` and `example/Rac_lm.jl` for complete working examples.
