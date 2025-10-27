# API Reference

This concise reference mirrors Kore’s command tables. It focuses on the exported functions and structs you will call from scripts or notebooks. For implementation details, inspect the source files noted in the rightmost column.

## Core Types

| Type | Description | Defined in |
| --- | --- | --- |
| `ChebyshevDiffn` | Chebyshev differentiation matrices and collocation grid | `src/Chebyshev.jl` |
| `OnsetParams` | Parameter set for single-mode stability analysis | `src/linear_stability.jl` |
| `ShellParams` | Helper constructor for `OnsetParams` | `src/linear_stability.jl` |
| `LinearStabilityOperator` | Sparse operator assembled from `OnsetParams` | `src/linear_stability.jl` |
| `BasicState` | Axisymmetric base state coefficients | `src/basic_state.jl` |
| `BasicState3D` | 3-D base state coefficients | `src/basic_state.jl` |
| `TriGlobalParams` | Parameters for multi-mode coupled problems | `src/triglobal_stability.jl` |
| `CoupledModeProblem` | Book-keeping for tri-global matrices | `src/triglobal_stability.jl` |

## Solvers

| Function | Purpose | Key Keywords | Notes |
| --- | --- | --- | --- |
| `solve_eigenvalue_problem(A, B; kwargs...)` | Wrapper around ARPACK or KrylovKit | `solver`, `nev`, `which`, `tol`, `maxiter` | Accepts sparse or dense matrices. |
| `leading_modes(op; kwargs...)` | Compute leading eigenpairs of a linear stability operator | `nev`, `solver`, `which` | Operates on `LinearStabilityOperator` directly. |
| `find_growth_rate(op; kwargs...)` | Return the largest growth rate and associated eigenvector | `nev`, `which` | Handy for fixed Rayleigh numbers. |
| `find_critical_rayleigh(; kwargs...)` | Bracket search for critical `Ra` | `Ra_guess`, `solver`, `tolerance`, `max_iterations` | Returns `(Ra_c, ω_c, eigenvector)`. |
| `solve_triglobal_eigenvalue_problem(problem; kwargs...)` | Solve coupled-mode eigenproblem | `nev`, `solver`, `which`, `tol` | Works with `CoupledModeProblem`. |
| `find_critical_rayleigh_triglobal(; kwargs...)` | Experimental: critical `Ra` search in coupled-mode setting | `Ra_guess`, `m_range` | Requires `BasicState3D`. |

## Field Reconstruction

| Function | Purpose | Notes |
| --- | --- | --- |
| `potentials_to_velocity(op, poloidal, toroidal)` | Convert spectral potentials to velocity components on the radial grid. | Useful for verifying boundary conditions. |
| `velocity_fields_from_poloidal_toroidal(op, eigenvector; nθ, nφ)` | Evaluate velocity components in real space. | Provides `u_r`, `u_θ`, `u_φ` arrays. |
| `temperature_field_from_coefficients(op, eigenvector; nθ, nφ)` | Reconstruct temperature perturbation. | Shares the grid with velocity helper. |
| `fields_from_coefficients(op, eigenvector; nθ, nφ)` | Convenience wrapper returning all fields. | Returns a named tuple with grid metadata. |

## Tri-Global Utilities

| Function | Purpose | Notes |
| --- | --- | --- |
| `setup_coupled_mode_problem(params)` | Build `CoupledModeProblem` from `TriGlobalParams`. | Computes coupling graph and block indices. |
| `estimate_triglobal_problem_size(params)` | Predict matrix sizes and DoF counts. | Helps decide solver strategy. |
| `get_coupling_modes(m, m_bs, m_range)` | Identify which perturbation modes couple via base state mode `m_bs`. | Returns sorted vector of mode indices. |

## Miscellaneous

| Function | Purpose |
| --- | --- |
| `print_cross_header()` | Print the ASCII banner (see `src/banner.jl`). |
| `CROSS_BANNER` | Raw banner string for custom display. |

## Solver Options

`solve_eigenvalue_problem` accepts the following `solver` identifiers:

- `:arpack` (default) – Julia’s ARPACK wrapper, optionally with `arpack_shift`.
- `:krylov` – KrylovKit shift-invert fallback when ARPACK fails to converge.

Common keywords:

- `nev` – number of eigenpairs.
- `which` – selection criterion (`:LR`, `:LI`, `:LM`, `:SR`, etc.).
- `tol` – convergence tolerance (default `1e-8`).
- `maxiter` – maximum iterations (backend dependent).
- `arpack_shift` – optional complex shift targeting eigenvalues near the neutral curve.

## Logging Helpers

- Set `ENV["CROSS_VERBOSE"] = "1"` to print additional diagnostics.
- Use `ENV["MKL_DEBUG_CPU_TYPE"] = 5` (Intel) or `BLAS.set_num_threads` to control threading.

## Checklist

- [ ] Check the defined-in column before modifying internals.
- [ ] Prefer the high-level helpers unless you need custom assembly.
- [ ] When switching solvers, adjust `which` to match your physical objective.
