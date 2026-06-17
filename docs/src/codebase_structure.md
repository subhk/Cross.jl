# Codebase Structure

<div class="cross-hero">
  <div class="cross-eyebrow">For developers</div>
  <h1>How the Cross.jl source is organized.</h1>
  <p>An overview of the source-code layout and architecture to help you navigate and extend the codebase.</p>
</div>

## Directory Layout

```
Cross.jl/
├── src/                              # Source code
│   ├── Cross.jl                      # Main module — includes submodules, exports public API
│   ├── types.jl                      # v2.0: StabilityResult, problem types, estimate_size
│   ├── validation.jl                 # v2.0: Input validation with errors and warnings
│   ├── show.jl                       # v2.0: Pretty-printing for all public types
│   │
│   ├── Spectral/
│   │   ├── Spectral.jl               # Entry point — includes chebyshev + ultraspherical
│   │   ├── chebyshev.jl              # ChebyshevDiffn differentiation matrices
│   │   └── ultraspherical.jl        # Olver-Townsend sparse spectral method
│   │
│   ├── Operators/
│   │   ├── Operators.jl              # Entry point
│   │   ├── sparse_operator.jl       # Sparse hydrodynamic operators
│   │   └── boundary_conditions.jl   # Mechanical, thermal, magnetic BCs
│   │
│   ├── BasicStates/
│   │   ├── BasicStates.jl            # Entry point
│   │   ├── basic_state.jl           # BasicState, BasicState3D, SphericalHarmonicBC types
│   │   ├── advection_diffusion.jl   # Self-consistent solver
│   │   └── basic_state_operators.jl # Coupling operators
│   │
│   ├── Stability/
│   │   ├── Stability.jl              # Entry point
│   │   ├── linear.jl                # OnsetParams, LinearStabilityOperator
│   │   ├── solver.jl                # KrylovKit shift-invert eigensolvers
│   │   ├── velocity.jl              # Velocity reconstruction
│   │   ├── onset.jl                 # Onset convection (no mean flow)
│   │   ├── biglobal.jl              # Biglobal (axisymmetric mean flow)
│   │   └── triglobal.jl             # Triglobal (3D mode coupling)
│   │
│   └── MHD/
│       ├── MHD.jl                    # Entry point
│       ├── types.jl                  # MHDParams, BackgroundField enum
│       ├── dipole.jl                 # Dipole field operators
│       ├── operator_functions.jl    # Lorentz, induction, diffusion operators
│       └── assembly.jl              # MHD matrix assembly
│
├── ext/
│   ├── CrossRecipesBaseExt/          # Plots.jl recipes (weak dep)
│   └── CrossMakieExt/                # Makie visualization (weak dep)
│
├── test/                             # Test suite
├── example/                          # Example scripts
├── docs/                             # Documentation (MkDocs)
└── Project.toml                      # Julia package manifest
```

## Core Architecture

### Module Entry Point (`Cross.jl`)

The main module file orchestrates all submodules and exports the public API:

```julia
module Cross
    # Dependencies
    using LinearAlgebra, SparseArrays, JLD2, Printf
    using Parameters
    using KrylovKit

    # v2.0 core (order matters!)
    include("types.jl")                # StabilityResult, problem types, estimate_size
    include("validation.jl")           # Input validation
    include("show.jl")                 # Pretty-printing for public types

    # Submodules
    include("Spectral/Spectral.jl")        # Chebyshev + ultraspherical discretization
    include("Operators/Operators.jl")      # Sparse operators + boundary conditions
    include("BasicStates/BasicStates.jl")  # Basic state types and coupling operators
    include("Stability/Stability.jl")      # Eigenvalue machinery and analysis modes
    include("MHD/MHD.jl")                  # MHD extension

    export ...
end
```

### Three Analysis Modes

Cross.jl provides three distinct analysis modes for different physical scenarios:

| Mode | File | Basic State | Use Case |
|------|------|-------------|----------|
| **Onset** | `Stability/onset.jl` | None (conduction only) | Classical convection onset |
| **Biglobal** | `Stability/biglobal.jl` | Axisymmetric (``m=0``) | Thermal wind effects |
| **Triglobal** | `Stability/triglobal.jl` | Non-axisymmetric | 3D boundary forcing |

## Source File Descriptions

### v2.0 Core Files

#### `types.jl`
Defines the `StabilityResult` return type, common problem parameter types, and `estimate_size` utilities introduced in v2.0.

#### `validation.jl`
Input validation layer introduced in v2.0. Emits structured errors and warnings before problem setup to catch misconfigurations early.

#### `show.jl`
Pretty-printing methods (`Base.show`) for all public types, introduced in v2.0.

### Spectral Submodule (`Spectral/`)

#### `Spectral/chebyshev.jl`
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

#### `Spectral/ultraspherical.jl`
Olver-Townsend sparse spectral method using ultraspherical (Gegenbauer) polynomials for large-scale problems.

### Operators Submodule (`Operators/`)

#### `Operators/sparse_operator.jl`
Sparse hydrodynamic operators using the ultraspherical spectral basis.

#### `Operators/boundary_conditions.jl`
Mechanical, thermal, and magnetic boundary condition application.

### BasicStates Submodule (`BasicStates/`)

#### `BasicStates/basic_state.jl`
Definitions for background temperature and flow states, including the `SphericalHarmonicBC` type (v2.0).

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

#### `BasicStates/advection_diffusion.jl`
Self-consistent advection-diffusion solver for computing basic states.

**Key Functions:**
- `solve_thermal_wind_balance!(bs, E, Ra, Pr)` - Compute axisymmetric thermal wind
- `solve_thermal_wind_balance_3d!(bs3d, E, Ra, Pr)` - Compute 3D thermal wind

#### `BasicStates/basic_state_operators.jl`
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

### Stability Submodule (`Stability/`)

#### `Stability/linear.jl`
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

#### `Stability/solver.jl`
Eigenvalue solving via KrylovKit shift-invert iteration.

**Key Functions:**
- `solve_eigenvalue_problem(op; nev, which)` - Compute eigenvalues
- `find_critical_rayleigh(E, Pr, χ, m, lmax, Nr; tol)` - Find critical Ra

#### `Stability/velocity.jl`
Reconstruct velocity components from poloidal/toroidal potentials.

**Key Functions:**
- `potentials_to_velocity(P, T; Dr, Dθ, Lθ, r, sintheta, m)` - Grid-based velocity

### Analysis Modes

#### `Stability/onset.jl`
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

#### `Stability/biglobal.jl`
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

#### `Stability/triglobal.jl`
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

### MHD Submodule (`MHD/`)

#### `MHD/types.jl`
`MHDParams` parameter struct and `BackgroundField` enum for selecting the imposed magnetic field geometry.

#### `MHD/dipole.jl`
Dipole magnetic field operators for the background field.

#### `MHD/operator_functions.jl`
Lorentz force, induction, and magnetic diffusion operator terms.

#### `MHD/assembly.jl`
MHD matrix assembly — adds magnetic terms to the A/B matrices produced by the Stability submodule.

### Extension Packages (`ext/`)

Visualization support is provided through Julia's extension mechanism (weak dependencies):

- **`CrossRecipesBaseExt/`** - Plots.jl recipes, loaded automatically when `RecipesBase` is available
- **`CrossMakieExt/`** - Interactive Makie visualization, loaded automatically when a Makie backend is available

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
│  • Chebyshev / ultraspherical radial discretization             │
│  • Spherical harmonic angular expansion                         │
│  • Boundary condition application                               │
│  • Basic state coupling operators (if applicable)               │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Generalized Eigenvalue Problem                  │
│                      A x = λ B x                                │
│  • KrylovKit shift-invert iteration                             │
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

```math
\cos\theta \frac{\partial \bar{u}_\phi}{\partial r} - \frac{\sin\theta}{r} \bar{u}_\phi = -\frac{Ra \cdot E^2}{2 Pr \cdot r_o} \frac{\partial \bar{\Theta}}{\partial \theta}
```

**Implementation:** `solve_thermal_wind_balance!()` in `BasicStates/basic_state.jl`

1. Project temperature gradient onto spherical harmonics
2. Apply coupling coefficients (``\ell \to \ell \pm 1``)
3. Integrate ODE for each velocity mode
4. Apply boundary conditions (no-slip or stress-free)

### Mode Coupling (Triglobal)

Non-axisymmetric basic states couple perturbation modes:

```math
Y_{\ell_1, m_1} \times Y_{\ell_2, m_2} = \sum_{\ell'} G_{\ell_1 \ell_2 \ell'}^{m_1 m_2 m'} Y_{\ell', m_1+m_2}
```

**Implementation:** `setup_coupled_mode_problem()` in `Stability/triglobal.jl`

1. Identify non-zero ``m_{bs}`` modes in basic state
2. Build coupling graph: ``m \leftrightarrow m \pm m_{bs}``
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

| Package | Type | Purpose |
|---------|------|---------|
| `LinearAlgebra` | stdlib | Standard linear algebra |
| `SparseArrays` | stdlib | Sparse matrix support |
| `KrylovKit` | direct | Iterative eigensolvers |
| `Parameters` | direct | `@with_kw` struct macros |
| `JLD2` | direct | Data serialization |
| `WignerSymbols` | direct | Gaunt coefficients for spherical harmonic coupling |
| `SpecialFunctions` | direct | Special mathematical functions |
| `LinearMaps` | direct | Linear operator abstractions |
| `BenchmarkTools` | direct | Performance benchmarking |
| `RecipesBase` | weak | Plots.jl plot recipes (`CrossRecipesBaseExt`) |
| `Makie` | weak | Interactive visualization (`CrossMakieExt`) |

## Extension Points

### Adding New Basic States

1. Define new function in `BasicStates/basic_state.jl` returning `BasicState` or `BasicState3D`
2. Populate coefficient dictionaries for temperature and velocity
3. Ensure derivatives are computed consistently

### Adding New Physics

1. Create a new submodule directory under `src/` (e.g., `src/MyPhysics/`)
2. Define parameter struct with required fields
3. Implement matrix assembly functions
4. Add operators to the A/B matrices in assembly
5. Include the submodule entry point in `Cross.jl` and export as needed

### Custom Boundary Conditions

1. Extend `Operators/boundary_conditions.jl` with new BC type
2. Implement row replacement in `apply_boundary_conditions!()`
3. Add option to parameter structs
