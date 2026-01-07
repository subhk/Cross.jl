# Examples

Cross.jl includes several ready-to-run example scripts in the `example/` directory. This page provides an overview of each script and what you can learn from it.

## Running Examples

All examples should be run from the repository root with the project environment:

```bash
julia --project=. example/<script_name>.jl
```

Or from the Julia REPL:

```julia
using Pkg
Pkg.activate(".")
include("example/<script_name>.jl")
```

---

## Linear Stability Demo

**File:** `example/linear_stability_demo.jl`

**Purpose:** Basic demonstration of the linear stability solver.

**What it does:**
- Loops over azimuthal wavenumbers $m = 1, \ldots, 20$
- Computes leading eigenvalues at fixed Rayleigh number
- Displays growth rates and frequencies

**Key concepts:**
- `ShellParams` configuration
- `leading_modes` function
- Eigenvalue interpretation

```julia
# Sample output
m    Re(λ₁)          Im(λ₁)          iterations
------------------------------------------------
 1  -1.23456e-02   5.67890e-01      24
 2  -8.76543e-03   6.12345e-01      28
...
```

**When to use:** First introduction to Cross.jl, verifying installation.

---

## Critical Rayleigh Number Scan

**File:** `example/Rac_lm.jl`

**Purpose:** Find critical Rayleigh numbers across azimuthal modes.

**What it does:**
- Scans $m$ values to find $Ra_c(m)$
- Uses bisection to find where growth rate = 0
- Identifies the globally most unstable mode

**Key concepts:**
- `find_critical_rayleigh` function
- Parameter sweeps
- Critical mode identification

**Physical insight:** The critical Rayleigh number $Ra_c$ varies with $m$, and the minimum determines the first mode to become unstable.

---

## Basic State Onset

**File:** `example/basic_state_onset_example.jl`

**Purpose:** Demonstrate custom basic state usage.

**What it does:**
- Creates a conduction basic state
- Optionally adds meridional temperature variation
- Computes stability with the modified background

**Key concepts:**
- `ChebyshevDiffn` construction
- `conduction_basic_state` function
- Passing basic state to `ShellParams`

```julia
# Create basic state
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
bs = conduction_basic_state(cd, χ; lmax_bs=6)

# Use in onset calculation
params = ShellParams(..., basic_state=bs)
```

---

## Boundary-Driven Jet

**File:** `example/boundary_driven_jet.jl`

**Purpose:** Study stability with differential boundary rotation or heating.

**What it does:**
- Creates a basic state with boundary-driven flows
- Computes thermal wind from temperature gradients
- Analyzes modified stability

**Key concepts:**
- Thermal wind balance
- Boundary-driven circulation
- Flow-convection interaction

---

## Non-Axisymmetric Basic State

**File:** `example/nonaxisymmetric_basic_state.jl`

**Purpose:** Create 3D basic states for tri-global analysis.

**What it does:**
- Defines boundary mode amplitudes
- Constructs `BasicState3D`
- Prepares for tri-global coupling

**Key concepts:**
- $(l, m)$ indexed coefficients
- `nonaxisymmetric_basic_state` function
- Reality conditions

```julia
boundary_modes = Dict(
    (2, 0) => 0.1,    # Axisymmetric Y₂₀
    (2, 2) => 0.05,   # Non-axisymmetric Y₂₂
)

bs3d = nonaxisymmetric_basic_state(cd, χ, Ra, Pr;
    lmax_bs=8, mmax_bs=4, boundary_modes)
```

---

## Tri-Global Analysis Demo

**File:** `example/triglobal_analysis_demo.jl`

**Purpose:** Framework for mode-coupled stability problems.

**What it does:**
- Sets up `TriGlobalParams`
- Builds coupled mode problem
- Estimates problem size
- (Optional) Solves eigenvalue problem

**Key concepts:**
- Mode coupling through non-axisymmetric basic states
- Block matrix structure
- Size estimation before solving

**Note:** Tri-global problems can be very large. Start with small `m_range` and `lmax`.

---

## MHD Dynamo Example

**File:** `example/mhd_dynamo_example.jl`

**Purpose:** Complete MHD stability analysis workflow.

**What it does:**
- Defines `MHDParams` with magnetic field
- Builds `MHDStabilityOperator`
- Assembles matrices
- Solves eigenvalue problem
- Interprets results

**Key concepts:**
- Lehnert number and magnetic Prandtl number
- Background field types (axial, dipole)
- Magnetic boundary conditions

```julia
params = MHDParams(
    E = 1e-3, Pr = 1.0, Pm = 5.0,
    Ra = 1e4, Le = 0.1,           # Weak magnetic field
    ricb = 0.35, m = 2, lmax = 10, N = 16,
    B0_type = axial,
    bci = 1, bco = 1,             # No-slip
    bci_magnetic = 0, bco_magnetic = 0,  # Insulating
)

op = MHDStabilityOperator(params)
A, B, interior_dofs, _ = assemble_mhd_matrices(op)
```

---

## Thermal Wind Test

**File:** `example/test_thermal_wind.jl`

**Purpose:** Verify thermal wind balance implementation.

**What it does:**
- Creates temperature field with latitudinal variation
- Computes thermal wind from geostrophic balance
- Verifies consistency

**Key concepts:**
- Thermal wind equation
- Geostrophic balance
- Verification against analytic solutions

---

## Figure 2 Benchmark

**File:** `example/figure2_benchmark.jl`

**Purpose:** Reproduce published benchmark results.

**What it does:**
- Replicates parameters from Figure 2 of reference paper
- Computes critical curves
- Compares against published values

**Key concepts:**
- Benchmark validation
- Parameter matching
- Quantitative verification

---

## Quick Reference Table

| Script | Complexity | Compute Time | Key Learning |
|--------|------------|--------------|--------------|
| `linear_stability_demo.jl` | Beginner | ~1 min | Basic workflow |
| `Rac_lm.jl` | Beginner | ~5 min | Parameter sweeps |
| `basic_state_onset_example.jl` | Intermediate | ~2 min | Custom basic states |
| `boundary_driven_jet.jl` | Intermediate | ~3 min | Thermal wind |
| `nonaxisymmetric_basic_state.jl` | Intermediate | ~1 min | 3D states |
| `triglobal_analysis_demo.jl` | Advanced | ~10+ min | Mode coupling |
| `mhd_dynamo_example.jl` | Advanced | ~5 min | MHD physics |
| `test_thermal_wind.jl` | Intermediate | ~1 min | Verification |
| `figure2_benchmark.jl` | Intermediate | ~10 min | Benchmarking |

---

## Creating Your Own Scripts

Use this template for new analyses:

```julia
#!/usr/bin/env julia
# my_analysis.jl - Description

# Add Cross.jl to path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Cross
using Printf
using JLD2

# === Parameters ===
E = 1e-5
Pr = 1.0
Ra = 1e7
χ = 0.35
m = 10
lmax = 60
Nr = 64

# === Setup ===
params = ShellParams(
    E = E, Pr = Pr, Ra = Ra, χ = χ,
    m = m, lmax = lmax, Nr = Nr,
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)

# === Solve ===
println("Computing eigenvalues...")
eigenvalues, eigenvectors, op, info = leading_modes(params; nev=6)

# === Results ===
println("\nResults:")
for (i, λ) in enumerate(eigenvalues)
    @printf("  λ[%d] = %.6e + %.6ei\n", i, real(λ), imag(λ))
end

# === Save ===
@save "my_results.jld2" params eigenvalues eigenvectors
println("\nResults saved to my_results.jld2")
```

---

## See Also

- [Getting Started](getting_started.md) - Installation and setup
- [Problem Setup](problem_setup.md) - Detailed configuration guide
- [API Reference](reference.md) - Function documentation
