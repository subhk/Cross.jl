# API Reference

Complete reference for Cross.jl functions, types, and modules.

> **Note:** The public API is the v2 problem/solve/result interface. Lower-level
> operator constructors remain documented for advanced workflows.

---

## v2.0 Unified API

Cross.jl v2.0 introduces a unified problem/solve/result interface that covers all analysis
modes with a consistent workflow: construct a problem, call `solve`, inspect the result.

### Problem Types

```julia
# OnsetProblem -- onset convection (no mean flow)
problem = OnsetProblem(params)

# BiglobalProblem -- axisymmetric mean flow
problem = BiglobalProblem(params, basic_state)

# TriglobalProblem -- non-axisymmetric, coupled modes
problem = TriglobalProblem(params, basic_state_3d, m_range)

# MHDProblem -- magnetohydrodynamic
problem = MHDProblem(mhd_params)
problem = MHDProblem(mhd_params, basic_state)
```

---

### Unified Solve

```julia
result = solve(problem; nev=6, sigma=nothing)
# Returns StabilityResult with:
# result.eigenvalues      -- Vector{Complex{T}}
# result.eigenvectors     -- Matrix{Complex{T}}
# result.growth_rate      -- max real part
# result.frequency        -- imag part of fastest-growing
# result.problem          -- the problem that produced this
# result.extra            -- NamedTuple of analysis-specific data
```

---

### Convenience Accessors

```julia
growth_rate(result)    # same as result.growth_rate
frequency(result)      # same as result.frequency
leading_mode(result)   # eigenvector of fastest-growing mode
```

---

### Unified Basic State

```julia
bs = basic_state(params; mode=:conduction)
bs = basic_state(params; mode=:meridional, amplitude=0.05)
bs = basic_state(params; mode=:selfconsistent, max_iterations=50, tol=1e-10)
bs3d = basic_state(params; mode=:nonaxisymmetric, mmax_bs=2)
```

---

### Input Validation

`OnsetProblem`, `BiglobalProblem`, and `TriglobalProblem` constructors validate parameters
automatically.

**Hard errors** (constructor throws `ArgumentError`):

| Parameter | Constraint |
|-----------|------------|
| `chi` | Must be in (0, 1) |
| `E` | Must be > 0 |
| `Pr` | Must be > 0 |
| `Ra` | Must be >= 0 |
| `Nr` | Must be >= 8 |
| `lmax` | Must be >= 1 |
| `m` | Must be >= 0 |
| `mechanical_bc` | Must be a valid BC symbol (`:no_slip` or `:stress_free`) |
| `thermal_bc` | Must be a valid BC symbol (`:fixed_temperature` or `:fixed_flux`) |

**Warnings** (constructor prints to `@warn`):

- Low `Nr` (potential under-resolution)
- Extreme `E` values
- `lmax >> Nr` (spectral/radial mismatch)
- `m > lmax` (mode outside resolved range)

---

### Problem Size Estimation

```julia
estimate_size(problem)
# Prints matrix dimensions and estimated memory
# Auto-warns in solve() if > 8 GB
```

---

### Pretty-Printing

All v2.0 types (`OnsetProblem`, `BiglobalProblem`, `TriglobalProblem`, `MHDProblem`,
`StabilityResult`) have custom `show` methods for readable REPL output.

---

### Plot Extensions

**Plots.jl** (lightweight, static):

```julia
using Cross, Plots
plot(result)                          # eigenvalue spectrum
plot(results; sweep_param=:Ra)        # parameter sweep
```

**Makie** (interactive, publication-quality):

```julia
using Cross, CairoMakie
eigenspectrum(result)                 # interactive spectrum
plot_meridional(result, 1)            # meridional slice
plot_radial(result, 1)                # radial profiles
```

---

## Core Types

### Parameter Structures

#### `OnsetParams{T, BS}`

Parameter structure for onset problems. Use it directly with `OnsetProblem`, `BiglobalProblem`, or lower-level operator constructors.

```julia
@with_kw struct OnsetParams{T, BS}
    E::T                    # Ekman number
    Pr::T                   # Prandtl number
    Ra::T                   # Rayleigh number
    χ::T                    # Radius ratio r_i/r_o
    m::Int                  # Azimuthal wavenumber
    lmax::Int               # Maximum spherical harmonic degree
    Nr::Int                 # Radial resolution
    ri::T                   # Inner radius
    ro::T                   # Outer radius
    L::T                    # Gap width (ro - ri)
    mechanical_bc::Symbol   # :no_slip or :stress_free
    thermal_bc::Symbol      # :fixed_temperature or :fixed_flux
    use_sparse_weighting::Bool
    equatorial_symmetry::Symbol  # :both, :symmetric, or :antisymmetric
    basic_state::BS         # Optional BasicState or nothing
end
```

**Source:** `src/linear_stability.jl`

---

#### `TriglobalParams{T}`

Parameters for tri-global mode-coupled analysis.

```julia
@with_kw struct TriglobalParams{T}
    E::T
    Pr::T
    Ra::T
    χ::T
    m_range::UnitRange{Int}     # Range of coupled m modes
    lmax::Int
    Nr::Int
    basic_state_3d::BasicState3D
    mechanical_bc::Symbol
    thermal_bc::Symbol
end
```

**Source:** `src/triglobal_stability.jl`

### Operator Structures

#### `LinearStabilityOperator{T}`

Pre-assembled linear stability operator with cached matrices.

| Field | Type | Description |
|-------|------|-------------|
| `params` | `OnsetParams` | Problem parameters |
| `cd` | `ChebyshevDiffn` | Radial grid and derivative matrices |
| `r` | `Vector` | Radial collocation points |
| `index_map` | `Dict` | (ℓ, field) → index mapping |
| `l_sets` | `Dict` | ℓ mode sets by field type |
| `total_dof` | `Int` | Total degrees of freedom |
| `radial_cache` | `Dict` | Cached radial operators |

**Source:** `src/linear_stability.jl`

---

#### `CoupledModeProblem{T}`

Block-structured problem for tri-global analysis.

| Field | Type | Description |
|-------|------|-------------|
| `params` | `TriglobalParams` | Problem parameters |
| `m_range` | `UnitRange{Int}` | Coupled azimuthal modes |
| `coupling_graph` | `Dict{Int, Vector{Int}}` | Mode coupling structure |
| `all_m_bs` | `Vector{Int}` | Non-zero basic-state azimuthal modes |
| `block_indices` | `Dict{Int, UnitRange}` | Eigenvector index ranges per m |
| `total_dofs` | `Int` | Total degrees of freedom |

**Source:** `src/triglobal_stability.jl`

---


### Basic State Structures

#### `BasicState{T}`

Axisymmetric (m=0) basic state.

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

**Source:** `src/basic_state.jl`

---

#### `BasicState3D{T}`

Non-axisymmetric 3D basic state.

```julia
struct BasicState3D{T}
    lmax_bs::Int
    mmax_bs::Int
    Nr::Int
    r::Vector{T}
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    dtheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    ur_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    utheta_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    dur_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    dutheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    duphi_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
end
```

**Source:** `src/basic_state.jl`

---

#### `ChebyshevDiffn{T}`

Chebyshev differentiation matrices and grid.

```julia
struct ChebyshevDiffn{T<:AbstractFloat}
    n::Int              # Number of points
    domain::Tuple{T,T}  # Physical domain [a, b]
    max_order::Int      # Highest derivative order
    x::Vector{T}        # Collocation points
    D1::Matrix{T}       # First derivative
    D2::Matrix{T}       # Second derivative
    D3::Matrix{T}       # Third derivative (if computed)
    D4::Matrix{T}       # Fourth derivative (if computed)
end
```

**Constructor:**
```julia
cd = ChebyshevDiffn(N, [a, b], max_order)
```

**Source:** `src/Chebyshev.jl`

---

## Solver Functions

### `solve_eigenvalue_problem`

Solve the generalized eigenvalue problem $A\mathbf{x} = \sigma B\mathbf{x}$.

```julia
op = LinearStabilityOperator(params)
eigenvalues, eigenvectors, info = solve_eigenvalue_problem(
    op;
    nev = 6,              # Number of eigenvalues
    which = :LR,          # Selection: :LR, :LI, :LM
    tol = 1e-8,           # Convergence tolerance
    maxiter = 100,        # Maximum iterations
)
```

**Returns:**
- `eigenvalues::Vector{ComplexF64}` - Sorted eigenvalues
- `eigenvectors::Vector{Vector{ComplexF64}}` - Corresponding eigenvectors
- `info::Dict` - Solver diagnostics

**Source:** `src/linear_stability.jl`

---

### `find_growth_rate`

Find the growth rate at fixed parameters.

```julia
σ, ω, eigvec = find_growth_rate(
    op;                   # LinearStabilityOperator
    nev = 8,
    which = :LR,
)
```

**Source:** `src/linear_stability.jl`

---

### `find_critical_rayleigh`

Search for the critical Rayleigh number.

```julia
Ra_c, ω_c, eigvec = find_critical_rayleigh(
    E, Pr, χ, m, lmax, Nr;
    Ra_guess = 1e6,
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
    tol = 1e-4,
    max_iterations = 50,
)
```

**Source:** `src/linear_stability.jl`

---

## Tri-Global Functions

### `setup_coupled_mode_problem`

Build the coupled mode eigenvalue problem.

```julia
problem = setup_coupled_mode_problem(params::TriglobalParams)
```

**Returns:** `CoupledModeProblem`

**Source:** `src/triglobal_stability.jl`

---

### `estimate_triglobal_problem_size`

Estimate DOF and matrix size requirements.

```julia
report = estimate_triglobal_problem_size(params::TriglobalParams)
```

**Returns:** Named tuple with `total_dofs`, `matrix_size`, `num_modes`, `dofs_per_mode`

**Source:** `src/triglobal_stability.jl`

---

### `solve_triglobal_eigenvalue_problem`

Solve the tri-global eigenvalue problem.

```julia
eigenvalues, eigenvectors = solve_triglobal_eigenvalue_problem(
    params;
    nev = 12,
    σ_target = 0.0,
    verbose = true,
)
```

**Returns:** `eigenvalues` vector and `eigenvectors` matrix (columns are modes)

**Source:** `src/triglobal_stability.jl`

---

### `find_critical_rayleigh_triglobal`

Find critical Ra in mode-coupled setting.

```julia
Ra_c, ω_c, eigvec = find_critical_rayleigh_triglobal(
    E, Pr, χ, m_range, lmax, Nr, basic_state_3d;
    Ra_guess = 1e7,
    tol = 1e-3,
)
```

**Source:** `src/triglobal_stability.jl`

---

## Basic State Functions

### `conduction_basic_state`

Create pure conduction basic state.

```julia
# Fixed temperature at both boundaries (default)
bs = conduction_basic_state(cd, χ, lmax_bs)

# Fixed flux at outer boundary
bs = conduction_basic_state(cd, χ, lmax_bs;
                            thermal_bc = :fixed_flux,
                            outer_flux = -1.0)
```

**Source:** `src/basic_state.jl`

---

### `meridional_basic_state`

Create basic state with meridional temperature variation.

```julia
# Fixed temperature at outer boundary
bs = meridional_basic_state(cd, χ, E, Ra, Pr, lmax_bs, amplitude;
                            mechanical_bc = :no_slip,
                            thermal_bc = :fixed_temperature)

# Fixed flux at outer boundary
bs = meridional_basic_state(cd, χ, E, Ra, Pr, lmax_bs, 0.0;
                            mechanical_bc = :no_slip,
                            thermal_bc = :fixed_flux,
                            outer_flux_mean = -1.0,
                            outer_flux_Y20 = 0.1)
```

**Source:** `src/basic_state.jl`

---

### `nonaxisymmetric_basic_state`

Create 3D basic state with specified boundary modes.

```julia
# Fixed temperature at outer boundary
bs3d = nonaxisymmetric_basic_state(
    cd, χ, E, Ra, Pr, lmax_bs, mmax_bs, amplitudes;
    thermal_bc = :fixed_temperature
)

# Fixed flux at outer boundary
bs3d = nonaxisymmetric_basic_state(
    cd, χ, E, Ra, Pr, lmax_bs, mmax_bs, Dict{Tuple{Int,Int},Float64}();
    thermal_bc = :fixed_flux,
    outer_fluxes = Dict((0,0) => -1.0, (2,0) => 0.1)
)
```

**Source:** `src/basic_state.jl`

---

### `basic_state`

High-level convenience function that accepts symbolic spherical harmonic BCs.

```julia
# Pure conduction
bs = basic_state(cd, χ, E, Ra, Pr)

# With symbolic temperature BC
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=Y20(0.1))

# With symbolic flux BC
bs = basic_state(cd, χ, E, Ra, Pr; flux_bc=Y00(-1.0) + Y20(0.1))

# Full signature
basic_state(cd, χ, E, Ra, Pr;
            temperature_bc = nothing,
            flux_bc = nothing,
            mechanical_bc = :no_slip,
            lmax_bs = nothing)
```

**Automatic dispatch:**
- Returns `BasicState` for axisymmetric BCs (m=0 only)
- Returns `BasicState3D` for non-axisymmetric BCs (m≠0)

**Source:** `src/basic_state.jl`

---

### `basic_state_selfconsistent`

Self-consistent basic state solver that accounts for temperature advection in non-axisymmetric cases.

```julia
# Self-consistent solver for non-axisymmetric BCs
bs, info = basic_state_selfconsistent(cd, χ, E, Ra, Pr;
                                       temperature_bc=Y20(0.1) + Y22(0.05),
                                       verbose=true)

# Full signature
basic_state_selfconsistent(cd, χ, E, Ra, Pr;
                           temperature_bc = nothing,
                           flux_bc = nothing,
                           mechanical_bc = :no_slip,
                           lmax_bs = nothing,
                           max_iterations = 20,
                           tolerance = 1e-8,
                           verbose = false)
```

**Returns:** `(BasicState3D, ConvergenceInfo)` where `ConvergenceInfo` is a named tuple with:
- `iterations`: Number of iterations used
- `converged`: `true` if converged
- `residual_history`: Vector of residuals

**Note:** For axisymmetric BCs (m=0 only), falls back to standard solver since advection is zero.

**Source:** `src/basic_state.jl`

---

### `nonaxisymmetric_basic_state_selfconsistent`

Low-level self-consistent solver for non-axisymmetric basic states.

```julia
bs, info = nonaxisymmetric_basic_state_selfconsistent(
    cd, χ, E, Ra, Pr, lmax_bs, mmax_bs, amplitudes;
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
    outer_fluxes = Dict{Tuple{Int,Int}, Float64}(),
    max_iterations = 20,
    tolerance = 1e-8,
    verbose = false
)
```

**Source:** `src/basic_state.jl`

---

### `solve_poisson_mode`

Solve the radial Poisson equation for a single spherical harmonic mode.

```julia
T_lm, dT_dr = solve_poisson_mode(ℓ, m, r, D2, D1, r_i, r_o, forcing;
                                  inner_value = 0.0,
                                  outer_value = 0.0,
                                  outer_bc = :fixed_temperature,
                                  inner_bc = :fixed_temperature)
```

Solves: $\nabla^2 \bar{T}_{\ell m} = f_{\ell m}(r)$

**Source:** `src/basic_state.jl`

---

### `compute_phi_advection_spectral`

Compute the φ-advection term in spectral space for the advection-diffusion solver.

```julia
forcing = compute_phi_advection_spectral(theta_coeffs, uphi_coeffs, lmax_bs, mmax_bs, r)
```

**Source:** `src/basic_state.jl`

---

## Symbolic Spherical Harmonic BCs

### `SphericalHarmonicBC{T}`

Type representing boundary conditions in spherical harmonic expansion.

```julia
struct SphericalHarmonicBC{T<:Real}
    coeffs::Dict{Tuple{Int,Int}, T}
end
```

**Supported operators:** `+`, `-`, `*`, `/`

**Source:** `src/basic_state.jl`

---

### Harmonic Constructors

| Function | Description |
|----------|-------------|
| `Ylm(ℓ, m, amp)` | General spherical harmonic |
| `Y00(amp)` | Monopole (uniform) |
| `Y10(amp)` | Axial dipole |
| `Y11(amp)` | Equatorial dipole |
| `Y20(amp)` | Quadrupole (equator-pole) |
| `Y21(amp)` | Tesseral quadrupole |
| `Y22(amp)` | Sectoral quadrupole |
| `Y30(amp)` - `Y44(amp)` | Higher orders |

**Example:**
```julia
# Combined pattern
bc = Y20(0.1) + Y22(0.05) + 0.5 * Y40(0.02)

# Use with basic_state
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)
```

**Source:** `src/basic_state.jl`

---

### Utility Functions

| Function | Description |
|----------|-------------|
| `to_dict(bc)` | Convert to `Dict{Tuple{Int,Int}, T}` |
| `get_lmax(bc)` | Maximum ℓ in BC |
| `get_mmax(bc)` | Maximum m in BC |
| `get_lmax_mmax(bc)` | Tuple (lmax, mmax) |
| `is_axisymmetric(bc)` | True if m=0 only |

**Source:** `src/basic_state.jl`

---

## Field Reconstruction

### `potentials_to_velocity`

Convert poloidal/toroidal potentials to velocity.

```julia
u_r, u_θ, u_φ = potentials_to_velocity(
    P, T;
    Dr = Dr,
    Dθ = Dθ,
    Lθ = Lθ,
    r = r,
    sintheta = sintheta,
    m = m,
)
```

**Source:** `src/get_velocity.jl`

---

### Eigenvector slicing

`LinearStabilityOperator` stores spectral coefficients by `(ℓ, field)` in
`op.index_map`. Eigenvectors returned by `solve_eigenvalue_problem` are full
vectors that can be sliced into per-ℓ radial coefficients:

```julia
eigvec = eigenvectors[1]

P_coeffs = Dict{Int, Vector{ComplexF64}}()
T_coeffs = Dict{Int, Vector{ComplexF64}}()
Θ_coeffs = Dict{Int, Vector{ComplexF64}}()

for ℓ in op.l_sets[:P]
    P_coeffs[ℓ] = eigvec[op.index_map[(ℓ, :P)]]
end
for ℓ in op.l_sets[:T]
    T_coeffs[ℓ] = eigvec[op.index_map[(ℓ, :T)]]
end
for ℓ in op.l_sets[:Θ]
    Θ_coeffs[ℓ] = eigvec[op.index_map[(ℓ, :Θ)]]
end
```

## Eigenvalue Selection Options

The `which` parameter controls eigenvalue selection:

| Value | Description |
|-------|-------------|
| `:LR` | Largest real part (onset/instability) |
| `:SR` | Smallest real part |
| `:LM` | Largest magnitude |
| `:SM` | Smallest magnitude |
| `:LI` | Largest imaginary part |
| `:SI` | Smallest imaginary part |

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CROSS_VERBOSE` | Enable verbose output | `"0"` |
| `CROSS_THETA_POINTS` | Default meridional resolution | `"96"` |

---

## Module Structure

```
Cross (main module)
│
├── src/
│   ├── Cross.jl                    # Top-level module & exports
│   ├── Chebyshev.jl                # ChebyshevDiffn radial discretization
│   ├── basic_state.jl              # BasicState, BasicState3D, SphericalHarmonicBC
│   ├── get_velocity.jl             # Velocity reconstruction
│   ├── linear_stability.jl         # OnsetParams and dense onset solver
│   ├── triglobal_stability.jl      # TriglobalParams, CoupledModeProblem, solver
│   │
│   ├── problems/                   # v2.0 unified problem types
│   │   ├── onset.jl                # OnsetProblem
│   │   ├── biglobal.jl             # BiglobalProblem
│   │   ├── triglobal.jl            # TriglobalProblem
│   │   └── mhd.jl                  # MHDProblem
│   │
│   ├── solve/                      # v2.0 unified solve dispatch
│   │   └── solve.jl                # solve(problem; ...) -> StabilityResult
│   │
│   ├── result/                     # v2.0 result types & accessors
│   │   └── stability_result.jl     # StabilityResult, growth_rate, frequency, leading_mode
│   │
│   └── validation/                 # v2.0 input validation & size estimation
│       └── validate.jl             # Parameter checks, estimate_size
│
├── v2.0 Unified API
│   ├── OnsetProblem               # Onset convection (no mean flow)
│   ├── BiglobalProblem            # Axisymmetric mean flow
│   ├── TriglobalProblem           # Non-axisymmetric, coupled modes
│   ├── MHDProblem                 # Magnetohydrodynamic stability
│   ├── solve(problem; ...)        # Unified solver -> StabilityResult
│   ├── growth_rate, frequency     # Convenience accessors
│   ├── leading_mode               # Fastest-growing eigenvector
│   └── estimate_size              # Memory / size estimation
│
├── Lower-level API
│   ├── Core Types
│   │   ├── ChebyshevDiffn
│   │   ├── OnsetParams
│   │   ├── LinearStabilityOperator
│   │   ├── BasicState, BasicState3D
│   │   ├── SphericalHarmonicBC
│   │   ├── TriglobalParams
│   │   └── CoupledModeProblem
│   │
│   ├── Symbolic BC Constructors
│   │   ├── Ylm(l, m, amp)
│   │   ├── Y00, Y10, Y11
│   │   ├── Y20, Y21, Y22
│   │   └── Y30-Y44
│   │
│   └── Exported Functions
│       ├── solve_eigenvalue_problem
│       ├── find_growth_rate
│       ├── find_critical_rayleigh
│       ├── basic_state
│       ├── solve_triglobal_eigenvalue_problem
│       └── ... (see exports in Cross.jl)
│
├── Analysis Modes
│   ├── Onset Convection           # No mean flow
│   ├── Biglobal Stability         # Axisymmetric mean flow
│   ├── Triglobal Stability        # Non-axisymmetric mean flow
│   └── MHD Stability              # Magnetohydrodynamic (v2.0)
│
└── Plot Extensions
    ├── Plots.jl recipes            # plot(result), sweep plots
    └── Makie extensions            # eigenspectrum, plot_meridional, plot_radial
```

---

## See Also

- [Problem Setup](problem_setup.md) - Tutorial for first problems
- [Basic States](basic_states.md) - Detailed basic state usage
- [Tri-Global Analysis](triglobal.md) - Mode coupling details
