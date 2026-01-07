# Tri-Global Instability Analysis

Tri-global analysis captures mode coupling across multiple azimuthal wavenumbers when the base state is fully 3-D. This section is inspired by Kore’s advanced topics chapters and shows how to size, assemble, and solve the coupled eigenvalue problem.

## When to Use Tri-Global Analysis

- Boundary forcing varies with longitude (e.g., hemispheric heating).
- Zonal jets introduce azimuthal shear that couples neighbouring modes.
- Magnetic fields or compositional variations inject `m ≠ 0` components into the base state.

If your base state is axisymmetric (`BasicState`), you can stay with the single-mode onset analysis described earlier.

## 1. Define the Parameter Set

```julia
using Cross

params_triglobal = TriGlobalParams(
    E = 1e-5,
    Pr = 1.0,
    Ra = 1.2e7,
    χ = 0.35,
    m_range = -2:2,             # Coupled perturbation modes
    lmax = 40,
    Nr = 64,
    basic_state_3d = bs3d,      # Supplied BasicState3D
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)
```

`m_range` must be symmetric around the dominant perturbation mode to capture both forward and backward couplings.

## 2. Inspect the Coupling Graph

```julia
problem = setup_coupled_mode_problem(params_triglobal)

for (m, neighbors) in sort(problem.coupling_graph)
    println("Mode $m couples to $(join(neighbors, ", "))")
end
```

The coupling graph tells you which block matrices will be non-zero. It is computed from the non-zero `(ℓ, m_bs)` coefficients in the base state.

## 3. Estimate Matrix Size

```julia
size_report = estimate_triglobal_problem_size(params_triglobal)
@info "Tri-global footprint" size_report...
```

Use this report to choose solver settings and HPC resources. Tri-global problems grow rapidly with `lmax` and the number of coupled modes.

## 4. Assemble and Solve

```julia
solution = solve_triglobal_eigenvalue_problem(
    problem;
    nev = 12,
    solver = :arpack,
    which = :LR,   # largest real part
)

σ₁ = real(solution.values[1])
ω₁ = imag(solution.values[1])
@info "Leading tri-global mode" σ₁ ω₁
```

The returned object includes:

- `values` – eigenvalues sorted by the chosen criterion.
- `vectors` – full eigenvectors spanning all coupled modes.
- `metadata` – solver diagnostics (iterations, residual norms, etc.).

## 5. Post-Processing

Use `fields_from_coefficients` on each mode block to reconstruct the perturbation fields. The helper stores radial indices per `(ℓ, field)` pair in `problem.block_indices`.

```julia
function extract_mode(problem, eigenvector, mode)
    idx = problem.block_indices[mode]
    return eigenvector[idx]
end

mode0 = extract_mode(problem, solution.vectors[:, 1], 0)
```

## Tips

- Start with a narrow `m_range` (e.g., `-1:1`) and increase gradually.
- Use sparse storage for the base state dictionaries to keep the coupling graph tight.
- If ARPACK stalls, retry with `solver=:krylov` to fall back to the shift-invert Krylov solver.

## Checklist

- [ ] `TriGlobalParams` uses an m-range consistent with base state content.
- [ ] Coupling graph matches physical expectations (no missing neighbors).
- [ ] Estimated size is feasible for your hardware (consider memory per eigenpair).
- [ ] Solver converges within a reasonable iteration limit.
