# Codebase Structure

This page provides an overview of the Cross.jl source code organization, helping developers understand the architecture and navigate the codebase.

## Directory Layout

```
Cross.jl/
├── src/                    # Source code
│   │
│   │   # === INCLUDED IN CROSS MODULE ===
│   ├── Cross.jl            # Main module (entry point)
│   ├── Chebyshev.jl        # Radial discretization
│   ├── banner.jl           # ASCII banner and version info
│   ├── get_velocity.jl     # Field reconstruction
│   ├── basic_state.jl      # Basic state definitions
│   ├── basic_state_operators.jl  # Basic state operators
│   ├── thermal_wind.jl     # Thermal wind balance solver
│   ├── linear_stability.jl # Core stability analysis
│   ├── onset_convection.jl # Mode 1: No mean flow
│   ├── biglobal_stability.jl # Mode 2: Axisymmetric mean flow
│   ├── triglobal_stability.jl # Mode 3: Non-axisymmetric mean flow
│   │
│   │   # === STANDALONE FILES (NOT included in module) ===
│   ├── boundary_conditions.jl  # BC utilities (standalone)
│   ├── SparseOperator.jl       # Sparse ultraspherical method
│   ├── UltrasphericalSpectral.jl # Ultraspherical utilities
│   ├── OnsetEigenvalueSolver.jl  # Alternative eigensolver
│   ├── DipoleOperators.jl      # Dipole field operators
│   ├── MHDOperator.jl          # MHD extension
│   ├── MHDOperatorFunctions.jl # MHD operator terms
│   ├── MHDAssembly.jl          # MHD matrix assembly
│   └── CompleteMHD.jl          # MHD high-level interface
│
├── test/                   # Test suite
├── example/                # Example scripts
├── docs/                   # Documentation (MkDocs)
└── Project.toml            # Julia package manifest
```

## Core Architecture

### Module Entry Point (`Cross.jl`)

The main module file orchestrates all components:

```julia
module Cross
    # Dependencies
    using LinearAlgebra, SparseArrays, JLD2, Printf
    using Parameters
    using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM
    using KrylovKit

    # Core components (order matters!)
    include("Chebyshev.jl")           # Radial discretization
    include("banner.jl")               # ASCII banner
    include("get_velocity.jl")         # Field reconstruction
    include("basic_state.jl")          # Basic state types
    include("basic_state_operators.jl") # Basic state operators
    include("thermal_wind.jl")         # Thermal wind balance
    include("linear_stability.jl")     # Core eigenvalue machinery

    # Three analysis modes
    include("onset_convection.jl")     # No mean flow
    include("biglobal_stability.jl")   # Axisymmetric mean flow
    include("triglobal_stability.jl")  # Non-axisymmetric mean flow

    export ...
end
```

### Three Analysis Modes

Cross.jl provides three distinct analysis modes for different physical scenarios:

| Mode | File | Basic State | Use Case |
|------|------|-------------|----------|
| **Onset** | `onset_convection.jl` | None (conduction only) | Classical convection onset |
| **Biglobal** | `biglobal_stability.jl` | Axisymmetric ($m=0$) | Thermal wind effects |
| **Triglobal** | `triglobal_stability.jl` | Non-axisymmetric | 3D boundary forcing |

## Source File Descriptions

### Core Infrastructure

#### `Chebyshev.jl`
Chebyshev spectral differentiation for radial discretization.

**Key Types:**
```julia
struct ChebyshevDiffn{T<:AbstractFloat}
    n::Int            # Number of points
    domain::Tuple{T,T}
    max_order::Int
    x::Vector{T}      # Collocation points
    D1::Matrix{T}     # First derivative matrix
    D2::Matrix{T}     # Second derivative matrix
    D3::Matrix{T}     # Third derivative (if requested)
    D4::Matrix{T}     # Fourth derivative (if requested)
end
```

**Key Functions:**
- `ChebyshevDiffn(N, [r_i, r_o], nderiv)` - Construct differentiation matrices

#### `banner.jl`
ASCII banner and version information.

**Key Exports:**
- `CROSS_BANNER` - ASCII art banner string
- `print_cross_header()` - Print the banner

#### `thermal_wind.jl`
Thermal wind balance solver for computing zonal flow from temperature gradients.

**Key Functions:**
- `solve_thermal_wind_balance!(bs, E, Ra, Pr)` - Compute axisymmetric thermal wind
- `solve_thermal_wind_balance_3d!(bs3d, E, Ra, Pr)` - Compute 3D thermal wind
- `build_thermal_wind(cd, χ, E, Ra, Pr, lmax_bs, amplitude)` - Build thermal wind state

#### `linear_stability.jl`
Core linear stability analysis machinery shared by all modes.

**Key Types:**
```julia
struct OnsetParams{T<:Real}
    E::T              # Ekman number
    Pr::T             # Prandtl number
    Ra::T             # Rayleigh number
    χ::T              # Radius ratio
    m::Int            # Azimuthal wavenumber
    lmax::Int         # Maximum spherical harmonic degree
    Nr::Int           # Radial resolution
    ri::T             # Inner radius
    ro::T             # Outer radius
    L::T              # Gap width (ro - ri)
    mechanical_bc::Symbol
    thermal_bc::Symbol
    use_sparse_weighting::Bool
    equatorial_symmetry::Symbol
    basic_state
end

struct LinearStabilityOperator{T}
    params::OnsetParams{T}
    cd::ChebyshevDiffn{T}
    r::Vector{T}
    index_map::Dict{Tuple{Int,Symbol}, UnitRange{Int}}
    l_sets::Dict{Symbol, Vector{Int}}
    total_dof::Int
    radial_cache::Dict{Tuple{Int,Int}, Matrix{T}}
end
```

**Key Functions:**
- `assemble_matrices(op)` - Build A and B matrices
- `solve_eigenvalue_problem(op; nev, which)` - Compute eigenvalues
- `find_critical_rayleigh(E, Pr, χ, m, lmax, Nr; tol)` - Find critical Ra

#### `get_velocity.jl`
Convert poloidal/toroidal potentials on a grid to velocity components.

**Key Functions:**
- `potentials_to_velocity(P, T; Dr, Dθ, Lθ, r, sintheta, m)` - Grid-based velocity

### Basic States

#### `basic_state.jl`
Definitions for background temperature and flow states.

**Key Types:**
```julia
# Axisymmetric basic state (m=0 modes only)
struct BasicState{T<:Real}
    lmax_bs::Int
    Nr::Int
    r::Vector{T}
    theta_coeffs::Dict{Int,Vector{T}}     # θ̄_ℓ0(r)
    uphi_coeffs::Dict{Int,Vector{T}}      # ū_φ,ℓ0(r)
    dtheta_dr_coeffs::Dict{Int,Vector{T}}
    duphi_dr_coeffs::Dict{Int,Vector{T}}
end

# Non-axisymmetric basic state (multiple m modes)
struct BasicState3D{T<:Real}
    lmax_bs::Int
    mmax_bs::Int
    Nr::Int
    r::Vector{T}
    theta_coeffs::Dict{Tuple{Int,Int},Vector{T}}  # θ̄_ℓm(r)
    dtheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    ur_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    utheta_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    uphi_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    dur_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    dutheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    duphi_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
end
```

**Key Functions:**
- `conduction_basic_state(cd, χ, lmax_bs)` - Pure conduction profile
- `meridional_basic_state(cd, χ, E, Ra, Pr, lmax_bs, amplitude)` - With thermal wind
- `nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, lmax_bs, mmax_bs, amplitudes)` - 3D state
- `solve_thermal_wind_balance!(...)` - Compute zonal flow from temperature
- `solve_thermal_wind_balance_3d!(...)` - 3D thermal wind

#### `basic_state_operators.jl`
Operators for incorporating basic state effects into stability analysis.

**Key Types:**
```julia
struct BasicStateOperators{T}
    # Advection operators: ū·∇u' and u'·∇ū
    advection_matrices::Dict
    # Temperature advection: ū·∇θ' and u'·∇θ̄
    thermal_advection_matrices::Dict
end
```

**Key Functions:**
- `build_basic_state_operators(bs, params)` - Construct operators
- `add_basic_state_operators!(A, B, ops)` - Add to stability matrices

### Analysis Modes

#### `onset_convection.jl`
Classical convection onset without mean flow.

**Key Types:**
```julia
struct OnsetConvectionParams{T<:Real}
    E::T, Pr::T, Ra::T, χ::T
    m::Int, lmax::Int, Nr::Int
    mechanical_bc::Symbol
    thermal_bc::Symbol
end
```

**Key Functions:**
- `solve_onset_problem(params)` - Solve eigenvalue problem
- `find_critical_Ra_onset(params)` - Find critical Rayleigh number
- `find_global_critical_onset(E, Pr, χ; m_range)` - Scan all m modes
- `estimate_onset_problem_size(params)` - Memory/size estimates
- `onset_scaling_laws(E)` - Asymptotic predictions

#### `biglobal_stability.jl`
Stability analysis with axisymmetric mean flow (thermal wind).

**Key Types:**
```julia
struct BiglobalParams{T<:Real}
    E::T, Pr::T, Ra::T, χ::T
    m::Int, lmax::Int, Nr::Int
    basic_state::BasicState{T}
    mechanical_bc::Symbol
    thermal_bc::Symbol
end
```

**Key Functions:**
- `create_conduction_basic_state(params)` - Conduction profile
- `create_thermal_wind_basic_state(params; amplitude)` - With zonal flow
- `solve_biglobal_problem(params)` - Solve with basic state
- `find_critical_Ra_biglobal(params)` - Critical Ra with mean flow
- `compare_onset_vs_biglobal(params)` - Compare to no-flow case
- `sweep_thermal_wind_amplitude(params, amplitudes)` - Parameter study

#### `triglobal_stability.jl`
Tri-global analysis with non-axisymmetric basic states.

**Key Types:**
```julia
struct TriglobalParams{T<:Real}
    E::T, Pr::T, Ra::T, χ::T
    m_range::UnitRange{Int}  # Coupled perturbation modes
    lmax::Int, Nr::Int
    basic_state_3d::BasicState3D{T}
    mechanical_bc::Symbol
    thermal_bc::Symbol
end

struct CoupledModeProblem{T}
    params::TriglobalParams{T}
    m_range::UnitRange{Int}
    all_m_bs::Vector{Int}
    coupling_graph::Dict{Int,Vector{Int}}
    block_indices::Dict{Int,UnitRange{Int}}
    total_dofs::Int
end
```

**Key Functions:**
- `setup_coupled_mode_problem(params)` - Build coupled problem structure
- `estimate_triglobal_problem_size(params)` - Size/memory estimates
- `solve_triglobal_eigenvalue_problem(params; nev, σ_target, verbose)` - Solve coupled system
- `find_critical_rayleigh_triglobal(params)` - Critical Ra for 3D forcing

### Standalone Files (Not Included in Cross Module)

!!! warning "Not Part of Main Module"
    The following files exist in `src/` but are **NOT included** in the Cross.jl module.
    They are standalone implementations for future integration, alternative methods, or separate use.

#### Sparse/Ultraspherical Methods

- **`SparseOperator.jl`** - Sparse ultraspherical spectral method for large problems
- **`UltrasphericalSpectral.jl`** - Utilities for ultraspherical (Gegenbauer) polynomial methods

#### Alternative Solvers

- **`OnsetEigenvalueSolver.jl`** - Alternative eigenvalue solver implementation
- **`boundary_conditions.jl`** - Standalone boundary condition utilities

#### MHD Extension

- **`MHDOperator.jl`** - Main MHD operator definitions
- **`MHDOperatorFunctions.jl`** - Individual operator terms for MHD equations
- **`MHDAssembly.jl`** - Matrix assembly for MHD problems
- **`CompleteMHD.jl`** - High-level MHD interface
- **`DipoleOperators.jl`** - Dipole magnetic field operators

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Parameters                            │
│  (E, Pr, Ra, χ, m, lmax, Nr, boundary conditions)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Analysis Mode Selection                      │
├─────────────────┬─────────────────────┬─────────────────────────┤
│  Onset          │  Biglobal           │  Triglobal              │
│  (no mean flow) │  (axisymmetric)     │  (non-axisymmetric)     │
└────────┬────────┴──────────┬──────────┴────────────┬────────────┘
         │                   │                        │
         ▼                   ▼                        ▼
┌────────────────┐  ┌────────────────┐      ┌────────────────────┐
│ OnsetParams    │  │ BiglobalParams │      │ TriglobalParams    │
│                │  │ + BasicState   │      │ + BasicState3D     │
└────────┬───────┘  └────────┬───────┘      └─────────┬──────────┘
         │                   │                        │
         └───────────────────┼────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Matrix Assembly                              │
│  • Chebyshev radial discretization                              │
│  • Spherical harmonic angular expansion                         │
│  • Boundary condition application                               │
│  • Basic state operators (if applicable)                        │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Generalized Eigenvalue Problem                  │
│                      A x = λ B x                                │
│  • ArnoldiMethod (shift-invert)                                 │
│  • KrylovKit (iterative)                                        │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Results                                   │
│  • Eigenvalues (growth rates, frequencies)                      │
│  • Eigenvectors (mode structure)                                │
│  • Field reconstruction (velocity, temperature, magnetic)       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Algorithms

### Thermal Wind Balance

The thermal wind equation connects temperature gradients to zonal flow:

$$
\cos\theta \frac{\partial \bar{u}_\phi}{\partial r} - \frac{\sin\theta}{r} \bar{u}_\phi = -\frac{Ra \cdot E^2}{2 Pr \cdot r_o} \frac{\partial \bar{\Theta}}{\partial \theta}
$$

**Implementation:** `solve_thermal_wind_balance!()` in `basic_state.jl`

1. Project temperature gradient onto spherical harmonics
2. Apply coupling coefficients ($\ell \to \ell \pm 1$)
3. Integrate ODE for each velocity mode
4. Apply boundary conditions (no-slip or stress-free)

### Mode Coupling (Triglobal)

Non-axisymmetric basic states couple perturbation modes:

$$
Y_{\ell_1, m_1} \times Y_{\ell_2, m_2} = \sum_{\ell'} G_{\ell_1 \ell_2 \ell'}^{m_1 m_2 m'} Y_{\ell', m_1+m_2}
$$

**Implementation:** `setup_coupled_mode_problem()` in `triglobal_stability.jl`

1. Identify non-zero $m_{bs}$ modes in basic state
2. Build coupling graph: $m \leftrightarrow m \pm m_{bs}$
3. Allocate block-sparse matrix structure
4. Assemble diagonal (single-mode) and off-diagonal (coupling) blocks

## Testing

```
test/
├── runtests.jl            # Test runner
├── boundary_conditions.jl # BC application tests
├── chebyshev.jl           # Chebyshev differentiation tests
├── sparse_operator.jl     # Sparse method tests
├── thermal_wind.jl        # Thermal wind balance tests
└── triglobal.jl           # Triglobal stability tests
```

Run tests with:
```julia
using Pkg
Pkg.test("Cross")
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `LinearAlgebra` | Standard linear algebra |
| `SparseArrays` | Sparse matrix support |
| `ArnoldiMethod` | Shift-invert eigensolvers |
| `KrylovKit` | Iterative eigensolvers |
| `Parameters` | `@with_kw` struct macros |
| `JLD2` | Data serialization |
| `WignerSymbols` | Gaunt coefficients for spherical harmonic coupling |
| `SpecialFunctions` | Special mathematical functions |
| `LinearMaps` | Linear operator abstractions |
| `BenchmarkTools` | Performance benchmarking |

## Extension Points

### Adding New Basic States

1. Define new function in `basic_state.jl` returning `BasicState` or `BasicState3D`
2. Populate coefficient dictionaries for temperature and velocity
3. Ensure derivatives are computed consistently

### Adding New Physics

1. Create new operator file (e.g., `MyPhysicsOperator.jl`)
2. Define parameter struct with required fields
3. Implement matrix assembly functions
4. Add operators to the A/B matrices in assembly
5. Export from `Cross.jl`

### Custom Boundary Conditions

1. Extend `boundary_conditions.jl` with new BC type
2. Implement row replacement in `apply_boundary_conditions!()`
3. Add option to parameter structs
