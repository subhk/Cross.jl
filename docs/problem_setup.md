# Setting Up Your First Problem

This walkthrough mirrors the hands-on tutorials in Kore’s docs. By the end you will:

1. Assemble a linear stability operator for a rotating spherical shell.
2. Search for the critical Rayleigh number.
3. Inspect the leading eigenmode structure.
4. Persist results for reuse.

All commands run inside the project environment (`julia --project=. …`).

## 1. Choose Physical and Numerical Parameters

The `ShellParams` helper converts legacy inputs (inner/outer radii) into a consistent `OnsetParams` struct.

```julia
using Cross

params = ShellParams(
    E = 3e-6,              # Ekman number
    Pr = 1.0,              # Prandtl number
    Ra = 5e6,              # Initial Rayleigh guess
    χ = 0.35,              # Radius ratio r_i / r_o
    m = 8,                 # Azimuthal wavenumber
    lmax = 80,             # Maximum spherical harmonic degree
    Nr = 96,               # Radial grid resolution
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
);
```

Key checks performed by the constructor:

- `χ` must lie in `(0, 1)`.
- `lmax ≥ m`.
- `Nr ≥ 4` to enforce tau boundary conditions.

## 2. Inspect the Operator Footprint

```julia
op = LinearStabilityOperator(params)
println("Total DoF: ", op.total_dof)
println("ℓ-sets: ", op.l_sets)
```

Internally, Cross.jl builds Chebyshev differentiation matrices (`ChebyshevDiffn`) and maps each `(ℓ, field)` combination to a contiguous block in the sparse system. This mapping is stored in `op.index_map` and is useful when extracting eigenvectors.

## 3. Find the Critical Rayleigh Number

Use `find_critical_rayleigh` to perform a bracket–search in Rayleigh number while computing the leading eigenvalue at each iteration.

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
    solver = :arpack,
    nev = 6,
)

@info "Critical Rayleigh number" Ra_c ω_c
```

Arguments of note:

- `solver` – choose `:arpack` (default) or `:krylov`.
- `nev` – number of eigenvalues to compute; the first (largest growth rate) guides the search.
- `equatorial_symmetry` – restricts the ℓ-spectrum when set to `:symmetric` or `:antisymmetric`.

## 4. Interpret the Eigenvector

Eigenvectors stack the poloidal, toroidal, and temperature coefficients for each spherical harmonic. Use the helper `fields_from_coefficients` to convert them back to physical space on a sparse grid for plotting:

```julia
using LinearAlgebra

fields = fields_from_coefficients(op, eigvec; nθ = 128, nφ = 256)
r_slice = fields.radius
θ_slice = fields.colatitude
temperature = fields.temperature_amplitude
```

You can now visualise `temperature` or the velocity components with your preferred plotting library (Makie, PyPlot, etc.).

## 5. Persist Results With JLD2

```julia
using JLD2
@save "outputs/onset_case1.jld2" params Ra_c ω_c eigvec
```

Reload later with `@load` and skip recomputing expensive eigenvalue problems.

## 6. Automate Parameter Sweeps

Wrap the steps above inside a loop to scan `m`, `Ra`, or boundary conditions. A minimal template looks like:

```julia
function sweep_modes(m_values; base = params)
    results = Dict{Int, NamedTuple}()
    for m in m_values
        current = base |> (; m, Ra = base.Ra)
        Ra_c, ω_c, vec = find_critical_rayleigh(; current..., solver = :arpack)
        results[m] = (Ra_c = Ra_c, ω_c = ω_c)
    end
    return results
end

sweep = sweep_modes(6:12)
```

The splat `(; current..., solver = …)` pattern leverages keyword destructuring for clarity.

## Checklist

- [ ] `ShellParams` constructed without assertion failures.
- [ ] Operator degrees of freedom (DoF) align with expectations.
- [ ] `find_critical_rayleigh` converges from your `Ra_guess`.
- [ ] Eigenvector converted back to physical fields without NaNs or Infs.
- [ ] Results saved for later reuse.

Continue with the next page to construct customised basic states.
