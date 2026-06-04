# Cross.jl

[![CI](https://github.com/subhk/Cross.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/subhk/Cross.jl/actions/workflows/ci.yml)
[![Documentation](https://github.com/subhk/Cross.jl/actions/workflows/docs.yml/badge.svg)](https://subhk.github.io/Cross.jl/)

**Cross.jl** is a Julia package for linear stability analysis of rotating convection and magnetohydrodynamic (MHD) flows in spherical shells. It provides spectral methods to solve eigenvalue problems arising in geophysical and astrophysical fluid dynamics.

It uses the Olver–Townsend ultraspherical (Gegenbauer) spectral method, which yields banded radial operators and 98–99% matrix sparsity, so high-resolution problems stay tractable while retaining spectral accuracy.

## Features

- **Ultra-sparse spectral discretization** — banded ultraspherical operators; 98–99% sparsity.
- **Three analysis modes** — onset convection, biglobal (axisymmetric mean flow), and triglobal (non-axisymmetric, mode-coupled) stability.
- **MHD extension** — magnetoconvection and kinematic-dynamo problems with `no_field`, `axial`, and `dipole` background fields.
- **Spurious-free eigenvalues** — a banded Galerkin (BC-recombined) discretization for the onset, hydro, and insulating-axial-MHD pencils eliminates the spurious-mode swarm produced by the tau method; results match the collocation onset benchmark to ~1e-12.
- **Unified solver API** — one `solve(problem)` entry point across all problem types, returning a `StabilityResult`.
- **Critical-parameter search** — automated bracketing for critical Rayleigh numbers.
- **Flexible basic states** — conductive, meridional, non-axisymmetric, and self-consistent (advection-balanced) states.

## Installation

Cross.jl is not in the General registry; install from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/Cross.jl")
```

Requires Julia 1.10 or newer.

## Quick Start

Onset of rotating convection — find the leading eigenvalues at fixed parameters:

```julia
using Cross

# Ekman, Prandtl, Rayleigh, radius ratio, azimuthal wavenumber, truncations
params = OnsetParams(E=1e-4, Pr=1.0, Ra=1e6, χ=0.35, m=4, lmax=30, Nr=64)

problem = OnsetProblem(params)
estimate_size(problem)          # check matrix size before solving
result = solve(problem; nev=6)

result.growth_rate              # leading growth rate
result.frequency                # drift frequency
result.eigenvalues              # full returned spectrum
```

Find the critical Rayleigh number for the onset of convection:

```julia
Ra_c = find_critical_Ra(OnsetProblem(params))
```

## Analysis Modes

| Mode | Problem type | Mean flow | Use when |
|------|--------------|-----------|----------|
| Onset convection | `OnsetProblem` | none (conductive) | fundamental onset, no background flow |
| Biglobal | `BiglobalProblem` | axisymmetric ($m=0$) | latitudinal structure, modes decoupled |
| Triglobal | `TriglobalProblem` | non-axisymmetric | longitudinal structure, modes coupled via Gaunt coefficients |
| MHD | `MHDProblem` | background magnetic field | magnetoconvection / kinematic dynamo |

Biglobal and triglobal analyses run on a basic state built with `basic_state`:

```julia
bs   = basic_state(params; mode=:meridional)        # axisymmetric → BiglobalProblem
bs3d = basic_state(params; mode=:nonaxisymmetric)   # 3-D → TriglobalProblem

result = solve(BiglobalProblem(params, bs))
```

`mode` accepts `:conduction`, `:meridional`, `:nonaxisymmetric`, and `:selfconsistent`.

## MHD Example

Magnetoconvection with an axial background field (insulating magnetic boundaries, the default, route through the spurious-free Galerkin solver):

```julia
using Cross

params = MHDParams(E=4.225e-4, Pr=1.0, Pm=1.0, Ra=55.905, ricb=0.35,
                   m=4, lmax=8, N=32,
                   B0_type=axial, B0_amplitude=1.0, Le=1e-3)

result = solve(MHDProblem(params))
result.growth_rate
```

A background field requires `Le > 0`. Dipole fields and perfectly-conducting / finite-conductivity magnetic boundaries are solved via the tau method.

### SLEPc eigensolver backend (optional)

KrylovKit is the default eigensolver. A SLEPc backend is available as an optional
extension for distributed (MPI) eigensolving. It requires a system PETSc **and**
SLEPc built with **complex scalars** (`--with-scalar-type=complex`) **and MUMPS**
(`--download-mumps`, for the parallel shift-invert factorization), plus MPI, with
`PETSC_DIR`, `PETSC_ARCH`, and `SLEPC_DIR` set.

Install the Julia wrappers:

```julia
julia> ] add PetscWrap SlepcWrap
```

In a driver script, initialize SLEPc once (passing the MUMPS shift-invert options),
solve with `backend=:slepc`, then finalize:

```julia
using Cross, PetscWrap, SlepcWrap

Cross.slepc_init!("-eps_gen_non_hermitian -st_type sinvert -st_pc_type lu " *
                  "-st_pc_factor_mat_solver_type mumps -eps_target_magnitude")

result = solve(problem; backend=:slepc)   # onset, biglobal, triglobal, MHD

# Eigenvalues are valid on every rank; eigenvectors are gathered to rank 0 only:
using PetscWrap: MPI
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    @show result.eigenvalues
    # use result.eigenvectors here
end

Cross.slepc_finalize!()
```

Launch across `N` ranks:

```
mpirun -n N julia --project=. driver.jl
```

Notes: the eigensolve (MUMPS factorization + Krylov iterations) is distributed over
`COMM_WORLD`; assembly is currently **replicated** (each rank builds the full matrix
and inserts only its owned rows). Eigenvectors are gathered to **rank 0** (workers
get empty eigenvectors); eigenvalues are identical on all ranks. Call `slepc_init!`
/`slepc_finalize!` **once per process** (not per solve). If the extension is not
loaded or `slepc_init!` was not called, `backend=:slepc` raises an actionable error.

## License

Cross.jl is released under the MIT License.
