# Cross.jl

[![CI](https://github.com/subhk/Cross.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/subhk/Cross.jl/actions/workflows/ci.yml)
[![Documentation](https://github.com/subhk/Cross.jl/actions/workflows/docs.yml/badge.svg)](https://subhk.github.io/Cross.jl/)

**Cross.jl** is a high-performance Julia package for **linear stability analysis of rotating convection and magnetohydrodynamic (MHD) flows in spherical shells**. It provides spectral methods to solve eigenvalue problems arising in geophysical and astrophysical fluid dynamics.

## Documentation

**[Full Documentation](https://subhk.github.io/Cross.jl/)** - Installation guide, tutorials, API reference, and theory

## Key Features

- **Spectral Accuracy**: Chebyshev polynomials (radial) + spherical harmonics (angular)
- **Sparse Operators**: Ultraspherical (Olver-Townsend) method for O(N) bandwidth matrices
- **Eigenvalue Solvers**: Arnoldi/Krylov methods via ArnoldiMethod.jl and KrylovKit.jl
- **MHD Extension**: Full magnetohydrodynamic stability with Lorentz force coupling
- **Tri-Global Analysis**: Mode-coupled instabilities with non-axisymmetric basic states
- **Flexible BCs**: No-slip/stress-free (mechanical), fixed temperature/flux (thermal)

## Physics

Cross.jl solves the linearized Boussinesq equations for thermal convection in a rotating spherical shell:

```
∂u/∂t + 2Ω×u = -∇p + E∇²u + Ra·E²/Pr · Θr̂     (Momentum)
∇·u = 0                                          (Continuity)
∂Θ/∂t + u·∇T₀ = (E/Pr)∇²Θ                       (Energy)
```

**Dimensionless Parameters:**
| Parameter | Definition | Physical Meaning |
|-----------|------------|------------------|
| Ekman (E) | ν/(ΩL²) | Viscous/Coriolis ratio |
| Prandtl (Pr) | ν/κ | Momentum/thermal diffusivity |
| Rayleigh (Ra) | αgΔTL³/(νκ) | Buoyancy forcing strength |
| Radius ratio (χ) | rᵢ/rₒ | Shell geometry |

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/Cross.jl.git")
```

**Dependencies**: SHTnsKit.jl (spherical harmonic transforms), KrylovKit.jl, ArnoldiMethod.jl

## Quick Start

```julia
using Cross

# Define parameters for onset of convection
params = ShellParams(
    E = 1e-4,           # Ekman number
    Pr = 1.0,           # Prandtl number
    Ra = 1e6,           # Rayleigh number
    χ = 0.35,           # Radius ratio (Earth-like)
    m = 10,             # Azimuthal wavenumber
    lmax = 30,          # Max spherical harmonic degree
    Nr = 32,            # Radial points
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)

# Compute leading eigenvalues
eigenvalues, eigenvectors, op, info = leading_modes(params; nev=6)

# Check stability: Re(λ) > 0 means unstable
println("Leading eigenvalue: ", eigenvalues[1])
println("Growth rate: ", real(eigenvalues[1]))
println("Frequency: ", imag(eigenvalues[1]))
```

## Finding Critical Rayleigh Number

```julia
# Search for Ra where growth rate = 0
Ra_c, ω_c, eigvec = find_critical_rayleigh(
    1e-4,    # E
    1.0,     # Pr
    0.35,    # χ
    10,      # m
    30,      # lmax
    32;      # Nr
    Ra_guess = 1e6,
)

println("Critical Rayleigh number: ", Ra_c)
println("Critical frequency: ", ω_c)
```

## Code Architecture

```
Cross.jl/
├── src/
│   ├── Cross.jl                 # Main module
│   ├── Chebyshev.jl             # Chebyshev differentiation matrices
│   ├── UltrasphericalSpectral.jl # Sparse radial operators
│   ├── linear_stability.jl      # Core eigenvalue problem
│   ├── triglobal_stability.jl   # Mode-coupled analysis
│   ├── basic_state.jl           # Background state definitions
│   ├── get_velocity.jl          # Field reconstruction
│   ├── MHDOperator.jl           # MHD extension
│   ├── MHDAssembly.jl           # MHD matrix assembly
│   └── boundary_conditions.jl   # BC implementation
├── example/                     # Ready-to-run scripts
├── test/                        # Test suite
└── docs/                        # MkDocs documentation
```

## Methodology

### Toroidal-Poloidal Decomposition

Solenoidal velocity field decomposed as:
```
u = ∇×∇×(Pr̂) + ∇×(Tr̂)
```
- **P**: Poloidal scalar (radial + meridional flow)
- **T**: Toroidal scalar (azimuthal flow)

### Spectral Discretization

- **Radial**: Chebyshev-Gauss-Lobatto collocation on mapped domain
- **Angular**: Spherical harmonic expansion Y_ℓ^m(θ,φ)
- **Eigenvalue**: Generalized problem Ax = σBx solved with shift-invert Arnoldi

### Sparse Ultraspherical Method

The Olver-Townsend ultraspherical approach yields **banded** differentiation matrices:
- Standard Chebyshev: O(N²) dense
- Ultraspherical: O(N) sparse with bandwidth ~O(derivative order)

## MHD Extension

For magnetohydrodynamic problems:

```julia
using Cross.CompleteMHD

params = MHDParams(
    E = 1e-4, Pr = 1.0, Pm = 5.0,  # Magnetic Prandtl
    Ra = 1e6, Le = 0.1,             # Lehnert number
    ricb = 0.35, m = 4, lmax = 20, N = 24,
    B0_type = :axial,               # Background field
    bci_magnetic = 2,               # Perfect conductor (inner)
    bco_magnetic = 0,               # Insulating (outer)
)

op = MHDStabilityOperator(params)
A, B, interior_dofs, _ = assemble_mhd_matrices(op)
```

## Examples

| Script | Description |
|--------|-------------|
| `linear_stability_demo.jl` | Basic eigenvalue computation |
| `Rac_lm.jl` | Critical Rayleigh number scan |
| `basic_state_onset_example.jl` | Custom background states |
| `triglobal_analysis_demo.jl` | Mode-coupled instabilities |
| `mhd_dynamo_example.jl` | MHD stability analysis |

Run examples:
```bash
julia --project=. example/linear_stability_demo.jl
```

## Performance

Typical problem sizes and timings (Apple M1):

| Resolution | DOFs | Matrix Assembly | Eigensolve (6 modes) |
|------------|------|-----------------|---------------------|
| lmax=30, Nr=32 | ~3,000 | 0.5s | 2s |
| lmax=60, Nr=64 | ~12,000 | 2s | 15s |
| lmax=100, Nr=96 | ~30,000 | 8s | 60s |

## References

1. Christensen, U.R. and Wicht, J. (2015). *Numerical Dynamo Simulations*. Treatise on Geophysics.
2. Olver, S. and Townsend, A. (2013). *A fast and well-conditioned spectral method*. SIAM Review.
3. Zhang, K. and Liao, X. (2017). *Theory and Modeling of Rotating Fluids*. Cambridge University Press.

## Contributing

Contributions welcome! Please open an issue or pull request on [GitHub](https://github.com/subhk/Cross.jl).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

[Subhajit Kar](mailto:subhajitkar19@gmail.com)
