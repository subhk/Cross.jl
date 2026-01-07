# Magnetohydrodynamic Extension

Cross.jl ships with an experimental magnetohydrodynamic (MHD) module located in `src/CompleteMHD.jl`. This mirrors the modular extension strategy used in Kore: base functionality remains clean while advanced physics lives in an opt-in component.

## Overview

- **Goal:** study the linear stability of conducting fluids under rotation, thermal gradients, and imposed magnetic fields.
- **Core file:** `src/CompleteMHD.jl` (re-exports helper modules from `MHDOperator.jl`, `MHDOperatorFunctions.jl`, and `MHDAssembly.jl`).
- **Status:** “Implementation complete, validation in progress” (see `MHD_README.md`).

## 1. Load the Module

```julia
include("src/CompleteMHD.jl")
using .CompleteMHD
```

The extension is namespaced to avoid polluting the main `Cross` module.

## 2. Define Parameters

```julia
params = MHDParams(
    E = 1e-3,
    Pr = 1.0,
    Pm = 5.0,
    Ra = 1e4,
    Le = 0.1,
    ricb = 0.35,
    m = 2,
    lmax = 20,
    N = 32,
    B0_type = axial,
    bci = 1, bco = 1,
    bci_magnetic = 0,
    bco_magnetic = 0,
)
```

Key additions relative to the hydrodynamic case:

- `Pm` – magnetic Prandtl number (ν/η).
- `Le` – Lehnert number, controls imposed background field strength.
- `B0_type` – choose between `axial` and other supported base fields.
- Magnetic boundary conditions (`bci_magnetic`, `bco_magnetic`) specify insulating vs conducting walls.

## 3. Assemble the Operator

```julia
op = MHDStabilityOperator(params)
A, B, interior_dofs, info = assemble_mhd_matrices(op)
```

`A` and `B` are sparse matrices forming the generalised eigenproblem `A x = λ B x`. `interior_dofs` holds the indices of physical degrees of freedom after applying boundary conditions.

## 4. Solve the Eigenproblem

```julia
using .OnsetEigenvalueSolver

A_int = A[interior_dofs, interior_dofs]
B_int = B[interior_dofs, interior_dofs]

result = solve_eigenvalue_problem(A_int, B_int; nev = 10, which = :LR)

σ = real(result.σ)
ω = imag(result.σ)
@info "Leading MHD mode" σ ω
```

## 5. Example Script

Run `example/mhd_dynamo_example.jl` for a templated workflow. It prints stability diagnostics and demonstrates how to vary `Le` to map the dynamo threshold.

## Development Notes

- The Lorentz force and induction operators live in `MHDOperatorFunctions.jl` and reuse the ultraspherical spectral machinery from the hydrodynamic solver.
- Matrix assembly relies on sparse storage to keep memory usage manageable.
- Validation is ongoing; consult `docs/MHD_IMPLEMENTATION.md` for derivations and cross-checks.

## Checklist

- [ ] `MHDParams` chosen with physically realistic `Pm` and `Le`.
- [ ] Background magnetic field matches the boundary conditions.
- [ ] Eigenproblem reduces to hydrodynamic case when `Le → 0`.
- [ ] Saved results include both hydrodynamic and magnetic components for post-processing.
