# Getting Started

This guide walks you through installing Cross.jl and running your first stability analysis. Follow the steps in order on Linux, macOS, or Windows with WSL.

## Prerequisites

### Required

- **Julia 1.10 or newer** - Cross.jl is developed and tested against Julia 1.10.x and 1.11.x
- **Git** - For cloning and pulling updates

### Optional but Recommended

- **Python 3.10+** - For building the MkDocs documentation locally
- **VS Code with Julia extension** - For a richer REPL and plot experience
- **C/Fortran toolchain** - For building dependencies such as MKL or FFTW if Julia requests them

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/subhk/Cross.jl.git
cd Cross.jl
```

### Step 2: Instantiate the Julia Environment

Open Julia inside the project folder and instantiate the dependencies:

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
```

The first run downloads packages including:

| Package | Purpose |
|---------|---------|
| `LinearAlgebra` | Standard Julia linear algebra |
| `SparseArrays` | Sparse matrix operations |
| `KrylovKit` | Krylov subspace eigenvalue solvers |
| `ArnoldiMethod` | Arnoldi iteration eigenvalue solver |
| `JLD2` | HDF5-based file I/O |
| `WignerSymbols` | Wigner 3j symbols for mode coupling |
| `Parameters` | Keyword argument macros |

!!! tip "Subsequent Sessions"
    Once instantiated, subsequent Julia sessions only need `Pkg.activate(".")` to use the pre-compiled dependencies.

### Step 3: Run the Test Suite

Before creating new problems, verify your installation passes the regression tests:

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.test()
```

The tests assemble small eigenvalue problems to verify that matrix blocks and solver wrappers agree with stored benchmarks.

### Step 4: Verify an Example Script

For a basic smoke test, run the linear stability demo:

```bash
julia --project=. example/linear_stability_demo.jl
```

You should see output similar to:

```
m    Re(λ₁)          Im(λ₁)          iterations
------------------------------------------------
 1  -1.23456e-02   5.67890e-01      24
 2  -8.76543e-03   6.12345e-01      28
...
```

## Package Structure

After installation, the project has the following structure:

```
Cross.jl/
├── src/                      # Source code
│   ├── Cross.jl              # Main module entry point
│   ├── Chebyshev.jl          # Chebyshev differentiation
│   ├── UltrasphericalSpectral.jl  # Ultraspherical spectral method
│   ├── linear_stability.jl   # Onset operator assembly
│   ├── basic_state.jl        # Basic state construction
│   ├── basic_state_operators.jl   # Basic state coupling operators
│   ├── triglobal_stability.jl    # Tri-global mode coupling
│   ├── get_velocity.jl       # Field reconstruction
│   ├── boundary_conditions.jl    # BC enforcement
│   ├── OnsetEigenvalueSolver.jl  # Eigenvalue solver interface
│   ├── MHDOperator.jl        # MHD operator structure
│   ├── MHDOperatorFunctions.jl   # MHD operator implementations
│   ├── MHDAssembly.jl        # MHD matrix assembly
│   ├── CompleteMHD.jl        # Complete MHD module
│   └── banner.jl             # ASCII banner
├── example/                  # Example scripts
│   ├── linear_stability_demo.jl
│   ├── mhd_dynamo_example.jl
│   ├── triglobal_analysis_demo.jl
│   ├── basic_state_onset_example.jl
│   └── ...
├── test/                     # Test suite
├── docs/                     # Documentation
├── Project.toml              # Package dependencies
└── Manifest.toml             # Locked dependency versions
```

## Julia Configuration

### Recommended Startup Configuration

Add the following to `~/.julia/config/startup.jl` for a better REPL experience:

```julia
atreplinit() do repl
    try
        @eval using Revise
    catch err
        @warn "Revise not available" err
    end
end
```

This enables [Revise.jl](https://github.com/timholy/Revise.jl) to pick up changes as you edit source files - essential for iterative development.

### Environment Variables

Cross.jl recognizes several environment variables:

| Variable | Purpose | Example |
|----------|---------|---------|
| `CROSS_VERBOSE` | Enable verbose output | `"1"` |
| `CROSS_THETA_POINTS` | Default meridional resolution | `"96"` |

## Building Documentation Locally

The documentation uses MkDocs Material. To preview locally:

### Step 1: Create Python Virtual Environment

```bash
cd Cross.jl
python -m venv .venv-docs
source .venv-docs/bin/activate  # On Windows: .venv-docs\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r docs/requirements.txt
```

### Step 3: Serve Documentation

```bash
mkdocs serve
```

Open `http://127.0.0.1:8000` in your browser to see the rendered site with live reload.

## Your First Calculation

Let's compute the growth rate for a rotating convection problem:

```julia
using Cross

# Define physical parameters
E = 1e-5       # Ekman number
Pr = 1.0       # Prandtl number
Ra = 2.1e7     # Rayleigh number
m = 10         # Azimuthal wavenumber

# Create parameter structure
params = ShellParams(
    E = E,
    Pr = Pr,
    Ra = Ra,
    m = m,
    lmax = 60,            # Max spherical harmonic degree
    Nr = 64,              # Radial resolution
    ri = 0.35,            # Inner radius
    ro = 1.0,             # Outer radius
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)

# Compute leading eigenvalues
eigenvalues, eigenvectors, _, info = leading_modes(params; nev=4)

# Display results
for (i, λ) in enumerate(eigenvalues)
    println("λ[$i] = $(real(λ)) + $(imag(λ))im")
end
```

### Understanding the Output

The eigenvalues $\lambda = \sigma + i\omega$ represent:

- **Growth rate ($\sigma$)**: Positive values indicate instability (convection onset)
- **Drift frequency ($\omega$)**: Rate at which the pattern rotates azimuthally

For Earth-like parameters at the onset of convection:

- $\sigma \approx 0$ (marginal stability)
- $\omega > 0$ (prograde drift with rotation)

## Troubleshooting

### Package Refuses to Precompile

```julia
julia> Pkg.update()
julia> Pkg.instantiate()
```

If issues persist, delete `Manifest.toml` and re-instantiate.

### Out-of-Memory Errors

Reduce `lmax` or `Nr` in the examples. Start with smaller values:

```julia
params = ShellParams(
    ...,
    lmax = 30,    # Reduced from 60
    Nr = 32,      # Reduced from 64
)
```

### Solver Doesn't Converge

Try adjusting solver parameters:

```julia
eigenvalues, eigenvectors, _, info = leading_modes(params;
    nev = 4,
    tol = 1e-5,       # Relaxed tolerance
    maxiter = 200,    # More iterations
)
```

### Julia Version Mismatch

Ensure you're using Julia 1.10 or newer:

```julia
julia> VERSION
v"1.10.0"
```

## Next Steps

Now that Cross.jl is installed and working, proceed to:

1. **[Setting Up Your First Problem](problem_setup.md)** - Learn to configure and solve onset problems
2. **[Basic States](basic_states.md)** - Create custom background temperature and flow profiles
3. **[Examples](examples.md)** - Explore the example scripts in the `example/` directory

---

!!! success "Installation Complete"
    You're ready to start computing convection onset in rotating spherical shells!
