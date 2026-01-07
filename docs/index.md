# Cross.jl User Guide

Cross.jl solves rotating convection and magnetohydrodynamic stability problems using high–order spectral methods. The package targets physicists and geophysicists who need to explore the onset of convection, compute growth rates, and study fully coupled tri-global instabilities.

This guide mirrors the structure of the Kore documentation so you can publish it with MkDocs Material on GitHub Pages. Each section walks through the workflow a new user follows: install Julia, configure a problem, run an eigenvalue search, and interpret the output.

## What You Can Do With Cross.jl

- Evaluate the onset of convection in rotating spherical shells with flexible mechanical and thermal boundary conditions.
- Build axisymmetric (`BasicState`) or fully 3-D (`BasicState3D`) basic states and assess their linear stability.
- Track leading eigenpairs with ARPACK or KrylovKit via a unified solver interface.
- Benchmark against published results using the provided example scripts and regression tests.
- Extend the computation to include magnetic fields (`CompleteMHD`) for kinematic dynamo studies.

## Architecture Snapshot

| Module | Purpose | Key Types |
| --- | --- | --- |
| `Cross.jl` | Package entry point | exports public API |
| `linear_stability.jl` | Onset operator assembly | `OnsetParams`, `LinearStabilityOperator` |
| `basic_state.jl` | Base state construction | `BasicState`, `BasicState3D` |
| `triglobal_stability.jl` | Coupled-mode analysis | `TriGlobalParams`, `CoupledModeProblem` |
| `CompleteMHD.jl` | Dynamo extension | `MHDParams`, `MHDStabilityOperator` |
| `example/` | Ready-to-run scripts | setup templates |

## How To Use This Guide

1. **Installation & Environment** – follow the supported Julia versions and package instantiation steps.
2. **Create Your First Problem** – assemble an onset problem, find critical Rayleigh numbers, and visualise eigenfunctions.
3. **Configure Custom Base States** – craft thermal-wind balanced backgrounds and reuse saved states.
4. **Tri-Global Extensions** – understand when mode coupling matters and how to size the eigenproblem.
5. **MHD Dynamo Module** – enable the magnetic extension and solve the full system.
6. **API Reference** – use concise summaries for scripting or notebook work.

Each page ends with checklists and troubleshooting tips so you can reproduce the workflow quickly on a new machine.

## Quick Example

```julia
julia> using Pkg; Pkg.activate("Cross")
     Activating project at `~/Cross.jl`

julia> using Cross

julia> params = ShellParams(
           E = 1e-5,
           Pr = 1.0,
           Ra = 1e7,
           m = 10,
           lmax = 60,
           Nr = 80,
           mechanical_bc = :no_slip,
           thermal_bc = :fixed_temperature,
       );

julia> op = LinearStabilityOperator(params);

julia> result = find_growth_rate(op; nev = 8)
Leading eigenvalues (λ = σ + iω)
σ₁ = 2.31e-04, ω₁ = 9.87e-01
σ₂ = -3.58e-04, ω₂ = 1.02e+00
```

Continue to the next page for detailed installation and environment instructions.
