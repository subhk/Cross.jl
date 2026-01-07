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
    ri::T                   # Inner radius
    ro::T                   # Outer radius
    m::Int                  # Azimuthal wavenumber
    lmax::Int               # Maximum spherical harmonic degree
    Nr::Int                 # Radial resolution
    mechanical_bc::Symbol   # :no_slip or :stress_free
    thermal_bc::Symbol      # :fixed_temperature or :fixed_flux
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
    basic_state = nothing,
)
```

**Source:** `src/linear_stability.jl`

---

#### `TriGlobalParams{T}`

Parameters for tri-global mode-coupled analysis.

```julia
@with_kw struct TriGlobalParams{T}
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

---

#### `MHDParams{T}`

Parameters for MHD stability problems.

```julia
@with_kw struct MHDParams{T}
    E::T                    # Ekman number
    Pr::T                   # Prandtl number
    Pm::T                   # Magnetic Prandtl number
    Ra::T                   # Rayleigh number
    Le::T                   # Lehnert number
    ricb::T                 # Inner core radius
    m::Int                  # Azimuthal wavenumber
    lmax::Int               # Maximum spherical harmonic degree
    N::Int                  # Radial resolution
    symm::Int               # Equatorial symmetry (1 or -1)
    B0_type::BackgroundField
    B0_amplitude::T
    bci::Int                # Inner velocity BC
    bco::Int                # Outer velocity BC
    bci_thermal::Int        # Inner thermal BC
    bco_thermal::Int        # Outer thermal BC
    bci_magnetic::Int       # Inner magnetic BC
    bco_magnetic::Int       # Outer magnetic BC
    heating::Symbol         # :differential or :internal
end
```

**Source:** `src/MHDOperator.jl`

---

### Operator Structures

#### `LinearStabilityOperator{T}`

Pre-assembled linear stability operator with cached matrices.

| Field | Type | Description |
|-------|------|-------------|
| `params` | `OnsetParams` | Problem parameters |
| `total_dof` | `Int` | Total degrees of freedom |
| `l_sets` | `Dict` | ℓ mode sets by field type |
| `index_map` | `Dict` | (ℓ, field) → index mapping |
| `A` | `SparseMatrixCSC` | Physics operator |
| `B` | `SparseMatrixCSC` | Mass operator |

**Source:** `src/linear_stability.jl`

---

#### `CoupledModeProblem{T}`

Block-structured problem for tri-global analysis.

| Field | Type | Description |
|-------|------|-------------|
| `params` | `TriGlobalParams` | Problem parameters |
| `coupling_graph` | `Dict{Int, Vector{Int}}` | Mode coupling structure |
| `block_indices` | `Dict{Int, UnitRange}` | Eigenvector index ranges per m |
| `A` | `SparseMatrixCSC` | Assembled A matrix |
| `B` | `SparseMatrixCSC` | Assembled B matrix |

**Source:** `src/triglobal_stability.jl`

---

#### `MHDStabilityOperator{T}`

MHD operator with precomputed radial matrices.

| Field | Type | Description |
|-------|------|-------------|
| `params` | `MHDParams` | Problem parameters |
| `radial_ops` | `Dict` | Cached $r^p d^n/dr^n$ operators |
| `ll_u, ll_v, ll_f, ll_g, ll_h` | `Vector{Int}` | ℓ modes per field |
| `matrix_size` | `Int` | Total matrix dimension |

**Source:** `src/MHDOperator.jl`

---

### Basic State Structures

#### `BasicState{T}`

Axisymmetric (m=0) basic state.

```julia
struct BasicState{T}
    r::Vector{T}
    Nr::Int
    lmax_bs::Int
    theta_coeffs::Dict{Int, Vector{T}}
    dtheta_dr_coeffs::Dict{Int, Vector{T}}
    uphi_coeffs::Dict{Int, Vector{T}}
    duphi_dr_coeffs::Dict{Int, Vector{T}}
end
```

**Source:** `src/basic_state.jl`

---

#### `BasicState3D{T}`

Non-axisymmetric 3D basic state.

```julia
struct BasicState3D{T}
    r::Vector{T}
    Nr::Int
    lmax_bs::Int
    mmax_bs::Int
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
struct ChebyshevDiffn{T}
    x::Vector{T}        # Collocation points
    N::Int              # Number of points
    D1::Matrix{T}       # First derivative
    D2::Matrix{T}       # Second derivative
    D3::Matrix{T}       # Third derivative (if computed)
    D4::Matrix{T}       # Fourth derivative (if computed)
    domain::Tuple{T,T}  # Physical domain [a, b]
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
eigenvalues, eigenvectors, info = solve_eigenvalue_problem(
    A, B;
    nev = 6,              # Number of eigenvalues
    which = :LR,          # Selection: :LR, :LM, :SR, :SM
    tol = 1e-8,           # Convergence tolerance
    maxiter = 100,        # Maximum iterations
    sigma = nothing,      # Shift for shift-invert
)
```

**Returns:**
- `eigenvalues::Vector{ComplexF64}` - Sorted eigenvalues
- `eigenvectors::Matrix{ComplexF64}` - Corresponding eigenvectors
- `info::Dict` - Solver diagnostics

**Source:** `src/OnsetEigenvalueSolver.jl`

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
problem = setup_coupled_mode_problem(params::TriGlobalParams)
```

**Returns:** `CoupledModeProblem`

**Source:** `src/triglobal_stability.jl`

---

### `estimate_triglobal_problem_size`

Estimate memory and DOF requirements.

```julia
report = estimate_triglobal_problem_size(params::TriGlobalParams)
```

**Returns:** Named tuple with `total_modes`, `total_dofs`, `matrix_size`, `memory_estimate_gb`

**Source:** `src/triglobal_stability.jl`

---

### `solve_triglobal_eigenvalue_problem`

Solve the tri-global eigenvalue problem.

```julia
solution = solve_triglobal_eigenvalue_problem(
    problem;
    nev = 12,
    which = :LR,
    tol = 1e-6,
)
```

**Returns:** Named tuple with `values`, `vectors`, `metadata`

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
    cd, χ, Ra, Pr;
    lmax_bs = 8,
    mmax_bs = 4,
    boundary_modes = Dict((2,2) => 0.1),
)
```

**Source:** `src/basic_state.jl`

---

## Field Reconstruction

### `potentials_to_velocity`

Convert poloidal/toroidal potentials to velocity.

```julia
u_r, u_θ, u_φ = potentials_to_velocity(op, poloidal, toroidal)
```

**Source:** `src/get_velocity.jl`

---

### `velocity_fields_from_poloidal_toroidal`

Full 3D velocity reconstruction with spherical harmonic synthesis.

```julia
u_r, u_θ, u_φ = velocity_fields_from_poloidal_toroidal(
    cfg, r, pol, tor;
    Dr = nothing,
    real_output = true,
)
```

**Source:** `src/get_velocity.jl`

---

### `temperature_field_from_coefficients`

Reconstruct 3D temperature field.

```julia
T = temperature_field_from_coefficients(cfg, r, Theta_coeffs; Dr=nothing)
```

**Source:** `src/get_velocity.jl`

---

### `fields_from_coefficients`

Combined field reconstruction returning named tuple.

```julia
fields = fields_from_coefficients(op, eigenvector; nθ=128, nφ=256)

# Access fields
fields.radius
fields.colatitude
fields.longitude
fields.u_r
fields.u_theta
fields.u_phi
fields.temperature_amplitude
```

**Source:** `src/get_velocity.jl`

---

## MHD Functions

### `MHDStabilityOperator`

Constructor for MHD operator.

```julia
op = MHDStabilityOperator(params::MHDParams)
```

**Source:** `src/MHDOperator.jl`

---

### `assemble_mhd_matrices`

Assemble MHD eigenvalue problem matrices.

```julia
A, B, interior_dofs, info = assemble_mhd_matrices(op::MHDStabilityOperator)
```

**Returns:**
- `A::SparseMatrixCSC` - Physics operator
- `B::SparseMatrixCSC` - Mass operator
- `interior_dofs::Vector{Int}` - Interior DOF indices
- `info::Dict` - Assembly diagnostics

**Source:** `src/MHDAssembly.jl`

---

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
| `MKL_DEBUG_CPU_TYPE` | Intel MKL optimization | - |
| `SHTNSKIT_PATH` | Custom SHTnsKit location | - |

---

## Module Structure

```
Cross.jl
├── Cross (main module)
│   ├── ChebyshevDiffn
│   ├── OnsetParams, ShellParams
│   ├── LinearStabilityOperator
│   ├── BasicState, BasicState3D
│   ├── TriGlobalParams, CoupledModeProblem
│   └── Exported functions
│
├── CompleteMHD (MHD extension)
│   ├── MHDParams
│   ├── MHDStabilityOperator
│   ├── BackgroundField enum
│   └── assemble_mhd_matrices
│
└── OnsetEigenvalueSolver
    └── solve_eigenvalue_problem
```

---

## See Also

- [Problem Setup](problem_setup.md) - Tutorial for first problems
- [Basic States](basic_states.md) - Detailed basic state usage
- [Tri-Global Analysis](triglobal.md) - Mode coupling details
- [MHD Extension](mhd_extension.md) - MHD module reference
