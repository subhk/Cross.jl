# API Reference

Complete reference for Cross.jl functions, types, and modules.

## Core Types

### Parameter Structures

#### `OnsetParams{T, BS}`

Internal parameter structure for onset problems. Use `ShellParams` for construction.

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

#### `ShellParams`

User-friendly constructor for `OnsetParams`.

```julia
params = ShellParams(
    E = 1e-5,
    Pr = 1.0,
    Ra = 1e7,
    m = 10,
    lmax = 60,
    Nr = 64;
    χ = 0.35,              # or provide ri, ro
    ri = 0.35,
    ro = 1.0,
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
    use_sparse_weighting = true,
    equatorial_symmetry = :both,
    basic_state = nothing,
)
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

### `leading_modes`

Compute leading eigenpairs for a parameter set.

```julia
eigenvalues, eigenvectors, op, info = leading_modes(
    params;               # ShellParams or OnsetParams
    nev = 6,
    which = :LR,
    tol = 1e-6,
    maxiter = 120,
    nθ = 96,              # Meridional grid points
)
```

**Source:** `src/linear_stability.jl`

---

### `find_growth_rate`

Find the growth rate at fixed parameters.

```julia
eigenvalues, eigenvectors, op, info = find_growth_rate(
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
bs = conduction_basic_state(cd::ChebyshevDiffn, χ; lmax_bs=6)
```

**Source:** `src/basic_state.jl`

---

### `meridional_basic_state`

Create basic state with meridional temperature variation.

```julia
bs = meridional_basic_state(
    cd, χ, E, Ra, Pr;
    lmax_bs = 6,
    amplitude = 0.05,
    mechanical_bc = :no_slip,
)
```

**Source:** `src/basic_state.jl`

---

### `nonaxisymmetric_basic_state`

Create 3D basic state with specified boundary modes.

```julia
bs3d = nonaxisymmetric_basic_state(
    cd, χ, E, Ra, Pr, 8, 4, Dict((2,2) => 0.1)
)
```

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

## Utility Functions

### `print_cross_header`

Print the ASCII banner.

```julia
print_cross_header()
```

**Source:** `src/banner.jl`

---

### `CROSS_BANNER`

Raw banner string for custom display.

```julia
println(CROSS_BANNER)
```

**Source:** `src/banner.jl`

---

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
├── Core Types
│   ├── ChebyshevDiffn         # Radial discretization
│   ├── OnsetParams, ShellParams
│   ├── LinearStabilityOperator
│   ├── BasicState, BasicState3D
│   ├── TriglobalParams
│   └── CoupledModeProblem
│
├── Analysis Modes
│   ├── Onset Convection       # No mean flow
│   ├── Biglobal Stability     # Axisymmetric mean flow
│   └── Triglobal Stability    # Non-axisymmetric mean flow
│
└── Exported Functions
    ├── solve_eigenvalue_problem
    ├── leading_modes, find_growth_rate
    ├── find_critical_rayleigh
    ├── solve_triglobal_eigenvalue_problem
    └── ... (see exports in Cross.jl)
```

---

## See Also

- [Problem Setup](problem_setup.md) - Tutorial for first problems
- [Basic States](basic_states.md) - Detailed basic state usage
- [Tri-Global Analysis](triglobal.md) - Mode coupling details
