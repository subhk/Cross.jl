# Cross.jl

**High-Performance Spectral Methods for Rotating Convection and MHD Stability in Spherical Shells**

---

Cross.jl is a Julia package for analyzing the onset of convection and magnetohydrodynamic (MHD) instabilities in rotating spherical shells. It uses ultra-sparse spectral methods based on the Olver-Townsend ultraspherical (Gegenbauer) approach to achieve exceptional computational efficiency while maintaining spectral accuracy.

## Key Features

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **Ultra-Sparse Spectral Methods**

    ---

    Achieves 98-99% matrix sparsity using the ultraspherical spectral method, enabling efficient computation even for high-resolution problems.

-   :material-earth:{ .lg .middle } **Geophysical Applications**

    ---

    Designed for planetary core dynamics, stellar convection, and laboratory rotating fluid experiments.

-   :material-magnet:{ .lg .middle } **MHD Extension**

    ---

    Full magnetohydrodynamic capability with support for axial and dipolar background magnetic fields.

-   :material-cube-outline:{ .lg .middle } **Tri-Global Analysis**

    ---

    Mode coupling analysis for non-axisymmetric basic states with multiple azimuthal wavenumbers.

</div>

## Three Analysis Modes

Cross.jl supports three progressively complex stability analysis approaches:

<div class="grid cards" markdown>

-   :material-waves:{ .lg .middle } **[Onset Convection (No Mean Flow)](analysis/onset_convection.md)**

    ---

    Classical linear stability analysis with a conductive temperature profile and zero background flow. Find critical Rayleigh numbers, azimuthal wavenumbers, and drift frequencies for the onset of thermal convection.

    **Use when**: Studying fundamental convection onset without pre-existing flows.

-   :material-rotate-orbit:{ .lg .middle } **[Biglobal (Axisymmetric Mean Flow)](analysis/biglobal_stability.md)**

    ---

    Stability analysis with axisymmetric ($m=0$) background states including thermal wind, differential rotation, and zonal jets. Each perturbation mode $m$ remains decoupled but is modified by the mean flow.

    **Use when**: Background has latitudinal variations but no longitudinal structure.

-   :material-sphere:{ .lg .middle } **[Triglobal (Non-Axisymmetric Mean Flow)](analysis/triglobal_stability.md)**

    ---

    Full 3-D stability analysis with non-axisymmetric basic states. Mode coupling links perturbations at different azimuthal wavenumbers through Gaunt coefficients derived from Wigner 3j symbols.

    **Use when**: Background has longitudinal variations (e.g., CMB heat flux heterogeneity).

</div>

## What You Can Do With Cross.jl

- **Convection Onset**: Evaluate the onset of convection in rotating spherical shells with flexible mechanical and thermal boundary conditions
- **Critical Rayleigh Numbers**: Find critical parameters using automated bracket search algorithms
- **Basic States**: Build axisymmetric (`BasicState`) or fully 3-D (`BasicState3D`) basic states and assess their linear stability
- **Eigenvalue Analysis**: Track leading eigenpairs with KrylovKit via a unified solver interface
- **MHD Stability**: Study magnetoconvection and kinematic dynamo problems with the `CompleteMHD` module
- **Benchmarking**: Validate against published results using provided example scripts

## Quick Start

```julia
using Pkg
Pkg.activate("Cross.jl")
Pkg.instantiate()

using Cross

# Define problem parameters
params = ShellParams(
    E = 1e-5,              # Ekman number
    Pr = 1.0,              # Prandtl number
    Ra = 2.1e7,            # Rayleigh number
    m = 10,                # Azimuthal wavenumber
    lmax = 60,             # Maximum spherical harmonic degree
    Nr = 64,               # Radial resolution
    ri = 0.35,             # Inner radius
    ro = 1.0,              # Outer radius
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)

# Build operator and find growth rates
op = LinearStabilityOperator(params)
eigenvalues, eigenvectors, _, info = leading_modes(params; nev=8)

# Display results
println("Leading eigenvalue: ", eigenvalues[1])
println("Growth rate: ", real(eigenvalues[1]))
println("Drift frequency: ", imag(eigenvalues[1]))
```

## Architecture Overview

Cross.jl is organized into modular components for flexibility and extensibility. See [Codebase Structure](codebase_structure.md) for full details.

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `Cross.jl` | Package entry point | Exports public API |
| `Chebyshev.jl` | Radial discretization | `ChebyshevDiffn` |
| `linear_stability.jl` | Core stability analysis | `OnsetParams`, `LinearStabilityOperator` |
| `basic_state.jl` | Base state construction | `BasicState`, `BasicState3D` |
| `onset_convection.jl` | Mode 1: No mean flow | `OnsetConvectionParams` |
| `biglobal_stability.jl` | Mode 2: Axisymmetric mean flow | `BiglobalParams` |
| `triglobal_stability.jl` | Mode 3: Non-axisymmetric | `TriglobalParams`, `CoupledModeProblem` |
| `SparseOperator.jl` | Sparse ultraspherical method | `SparseOnsetParams` |
| `CompleteMHD.jl` | MHD dynamo extension | `MHDParams`, `MHDStabilityOperator` |

## Physical Problem

Cross.jl solves the linearized equations for rotating convection in a spherical shell:

$$
\frac{\partial \mathbf{u}}{\partial t} + 2\boldsymbol{\Omega} \times \mathbf{u} = -\nabla p + E\nabla^2\mathbf{u} + \frac{Ra \cdot E^2}{Pr} \Theta \hat{\mathbf{r}}
$$

$$
\frac{\partial \Theta}{\partial t} + \mathbf{u} \cdot \nabla T_0 = \frac{E}{Pr} \nabla^2 \Theta
$$

$$
\nabla \cdot \mathbf{u} = 0
$$

The velocity field is decomposed into poloidal and toroidal components:

$$
\mathbf{u} = \nabla \times \nabla \times (P \hat{\mathbf{r}}) + \nabla \times (T \hat{\mathbf{r}})
$$

Fields are expanded in spherical harmonics $Y_\ell^m(\theta, \phi)$ and Chebyshev polynomials $T_n(r)$ for spectral accuracy.

## Dimensionless Parameters

| Parameter | Symbol | Definition | Typical Range |
|-----------|--------|------------|---------------|
| Ekman number | $E$ | $\nu/(\Omega L^2)$ | $10^{-3}$ - $10^{-7}$ |
| Prandtl number | $Pr$ | $\nu/\kappa$ | 0.1 - 10 |
| Rayleigh number | $Ra$ | $\alpha g \Delta T L^3/(\nu \kappa)$ | $10^3$ - $10^8$ |
| Radius ratio | $\chi$ | $r_i/r_o$ | 0.2 - 0.9 |

For MHD problems, additional parameters include:

| Parameter | Symbol | Definition |
|-----------|--------|------------|
| Magnetic Prandtl | $Pm$ | $\nu/\eta$ |
| Lehnert number | $Le$ | $B_0/(\sqrt{\mu\rho}\Omega L)$ |
| Magnetic Ekman | $E_m$ | $E/Pm$ |

## Performance Characteristics

The ultraspherical spectral method achieves remarkable sparsity:

| Resolution (Nr) | Dense Storage | Sparse Storage | Sparsity |
|-----------------|---------------|----------------|----------|
| 32 | 8 KB | ~0.2 KB | ~97% |
| 64 | 33 KB | ~1 KB | ~98% |
| 128 | 131 KB | ~4 KB | ~99% |

This translates to:
- **~20x speedup** for matrix-vector products
- **~50x memory reduction** for large problems
- **Linear scaling** with resolution for many operations

## Documentation Guide

### Getting Started
1. **[Installation](getting_started.md)** - Environment setup and first run
2. **[First Problem](problem_setup.md)** - Assemble an onset problem and find critical Rayleigh numbers
3. **[Examples](examples.md)** - Ready-to-run example scripts

### Analysis Modes
4. **[Onset Convection](analysis/onset_convection.md)** - Classical onset with no mean flow
5. **[Biglobal Stability](analysis/biglobal_stability.md)** - Axisymmetric mean flows (thermal wind)
6. **[Triglobal Stability](analysis/triglobal_stability.md)** - Non-axisymmetric mean flows with mode coupling

### Advanced Topics
7. **[Basic States](basic_states.md)** - Construct custom temperature and flow profiles
8. **[Tri-Global Analysis](triglobal.md)** - Technical details of mode coupling
9. **[MHD Extension](mhd_extension.md)** - Magnetic field effects and dynamo problems

### Reference
10. **[API Reference](reference.md)** - Complete function and type documentation
11. **[Codebase Structure](codebase_structure.md)** - Source code organization and architecture
12. **[FAQ](faq.md)** - Troubleshooting and common questions

## Requirements

- **Julia**: 1.10 or newer
- **Dependencies**: LinearAlgebra, SparseArrays, KrylovKit, ArnoldiMethod, JLD2, WignerSymbols

## Citation

If you use Cross.jl in your research, please cite:

```bibtex
@software{cross_jl,
  author = {Kar, Subhajit},
  title = {Cross.jl: Spectral Methods for Rotating Convection},
  url = {https://github.com/subhk/Cross.jl},
  version = {1.0.0},
  year = {2025}
}
```

## License

Cross.jl is released under the MIT License.

## Acknowledgments

Cross.jl builds upon the mathematical foundations established by:

- Olver & Townsend (2013) - Ultraspherical spectral methods
- Christensen & Wicht (2015) - Numerical dynamo simulations
- The Kore project - Python implementation reference

---

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **[Get Started](getting_started.md)**

    ---

    Installation and environment setup

-   :material-code-braces:{ .lg .middle } **[Examples](examples.md)**

    ---

    Ready-to-run example scripts

-   :material-api:{ .lg .middle } **[API Reference](reference.md)**

    ---

    Complete function documentation

-   :material-file-tree:{ .lg .middle } **[Codebase Structure](codebase_structure.md)**

    ---

    Source code organization for developers

</div>
