# MHD Implementation in Cross.jl

## Overview

This document describes the magnetohydrodynamic (MHD) implementation in Cross.jl, which extends the hydrodynamic convection onset solver to include magnetic field interactions.

**Status:** ⚠️ EXPERIMENTAL - Full implementation following Kore structure

**Date:** 2025-10-26

---

## Contents

1. [Physical Model](#physical-model)
2. [Mathematical Formulation](#mathematical-formulation)
3. [Implementation Structure](#implementation-structure)
4. [Usage Examples](#usage-examples)
5. [Benchmark Tests](#benchmark-tests)
6. [References](#references)

---

## Physical Model

### Governing Equations

The MHD equations in a rotating spherical shell:

**Momentum (Navier-Stokes + Lorentz):**
```
∂u/∂t + 2Ω×u = -∇p + E∇²u + Ra/Pr θr̂ + Le²(∇×B)×B₀
```

**Induction:**
```
∂B/∂t = ∇×(u×B₀) + Em∇²B
```

**Heat:**
```
∂θ/∂t + u·∇T₀ = Etherm∇²θ
```

**Incompressibility:**
```
∇·u = 0,  ∇·B = 0
```

### Non-dimensional Parameters

| Parameter | Symbol | Definition | Typical Range |
|-----------|--------|------------|---------------|
| Ekman number | E | ν/(ΩL²) | 10⁻³ - 10⁻⁷ |
| Prandtl | Pr | ν/κ | 0.1 - 10 |
| Magnetic Prandtl | Pm | ν/η | 0.1 - 10 |
| Rayleigh | Ra | αgΔTL³/(νκ) | 10³ - 10⁸ |
| Lehnert | Le | B₀/(√(μρ)ΩL) | 0.01 - 1 |
| Thermal Ekman | Etherm | E/Pr | - |
| Magnetic Ekman | Em | E/Pm | - |

### Field Decomposition

Following Kore, fields are decomposed into toroidal-poloidal forms:

**Velocity:**
```
u = ∇×(∇×(u_pol r̂)) + ∇×(u_tor r̂)
```

**Magnetic Field:**
```
B = ∇×(∇×(f_pol r̂)) + ∇×(g_tor r̂)
```

**Temperature:**
```
T = T₀(r) + θ(r,θ,φ,t)
```

Each scalar function is expanded in spherical harmonics Y_l^m(θ,φ) and discretized radially using ultraspherical (Gegenbauer) polynomials.

---

## Mathematical Formulation

### Eigenvalue Problem

The linear stability analysis leads to:
```
A x = λ B x
```

Where `x = [u_pol, u_tor, f_pol, g_tor, θ]` contains all perturbation fields.

### Matrix Structure

The matrices have block structure:

```
        u     v     f     g     h
    ┌─────────────────────────────┐
  u │ I+C  Coff  0    Lor   -B    │  (2curl NS + Lorentz)
    │                             │
  v │ Coff  I    Lor  0     0     │  (1curl NS + Lorentz)
    │                             │
  f │ Ind  Ind   I    0     0     │  (no-curl induction)
    │                             │
  g │ Ind  Ind   0    I     0     │  (1curl induction)
    │                             │
  h │ Adv  0     0    0     I     │  (heat equation)
    └─────────────────────────────┘
```

Legend:
- I: Time derivative (diagonal blocks in B matrix)
- C: Coriolis (diagonal and off-diagonal)
- Coff: Coriolis off-diagonal coupling
- Lor: Lorentz force
- Ind: Induction
- B: Buoyancy
- Adv: Thermal advection

### Key Couplings

1. **Magnetic → Velocity (Lorentz Force)**
   - Poloidal B → Poloidal u (via toroidal B component)
   - Toroidal B → Toroidal u (via poloidal B component)
   - Strength: Le²

2. **Velocity → Magnetic (Induction)**
   - Poloidal u → Poloidal B (stretching background field)
   - Toroidal u → Toroidal B (shearing background field)
   - Strength: Le

3. **Temperature → Velocity (Buoyancy)**
   - θ → Poloidal u
   - Strength: Ra/Pr

4. **Velocity → Temperature (Advection)**
   - Poloidal u → θ
   - Strength: 1

---

## Implementation Structure

### Files

```
src/
├── MHDOperator.jl              # Main MHD operator structure
├── MHDOperatorFunctions.jl     # Individual operator implementations
├── MHDAssembly.jl              # Matrix assembly
└── CompleteMHD.jl              # Complete module (use this)

example/
└── mhd_dynamo_example.jl       # Usage example

docs/
└── MHD_IMPLEMENTATION.md       # This file
```

### Key Data Structures

**`MHDParams`** - Physical and numerical parameters
- Contains all dimensionless numbers (E, Pr, Pm, Ra, Le)
- Geometry (ricb, m, lmax, N)
- Boundary conditions (velocity, temperature, magnetic)
- Background field type

**`MHDStabilityOperator`** - Pre-computed radial operators
- All velocity operators (r^k D^n)
- All magnetic field operators
- Background field operators h(r)
- Mode structure (ll_u, ll_v, ll_f, ll_g, ll_h)

**`BackgroundField`** - Enum for field types
- `no_field`: Kinematic dynamo
- `axial`: Uniform axial field
- `dipole`: Dipolar field (future)

### Core Functions

**Lorentz Force Operators:**
```julia
operator_lorentz_poloidal_diagonal(op, l, Le)
operator_lorentz_poloidal_offdiag(op, l, m, offset, Le)
operator_lorentz_toroidal(op, l, Le)
```

**Induction Operators:**
```julia
operator_induction_poloidal_from_u(op, l)
operator_induction_poloidal_from_v(op, l)
operator_induction_toroidal_from_u(op, l, m, offset)
operator_induction_toroidal_from_v(op, l)
```

**Magnetic Diffusion:**
```julia
operator_magnetic_diffusion_poloidal(op, l, Em)
operator_magnetic_diffusion_toroidal(op, l, Em)
```

**Assembly:**
```julia
assemble_mhd_matrices(op)  # Returns (A, B, interior_dofs, info)
```

---

## Usage Examples

### Basic Dynamo Stability

```julia
include("src/CompleteMHD.jl")
using .CompleteMHD

# Define parameters
params = MHDParams(
    E = 1e-3,
    Pr = 1.0,
    Pm = 5.0,
    Ra = 1e4,
    Le = 0.1,          # Weak background field
    ricb = 0.35,
    m = 2,
    lmax = 20,
    N = 32,
    B0_type = axial,
    bci = 1, bco = 1,  # No-slip
    bci_magnetic = 0,  # Insulating boundaries
    bco_magnetic = 0
)

# Build operator and assemble
op = MHDStabilityOperator(params)
A, B, interior_dofs, info = assemble_mhd_matrices(op)

# Solve eigenvalue problem
A_int = A[interior_dofs, interior_dofs]
B_int = B[interior_dofs, interior_dofs]
result = solve_eigenvalue_problem(A_int, B_int)

# Analyze results
σ = result.σ
ω = result.ω
if real(σ) > 0
    println("Unstable! Dynamo onset detected")
end
```

### Parameter Scan

```julia
# Scan Rayleigh number for fixed magnetic field
Le = 0.1
Ra_values = 10.0.^range(3, 6, length=20)

for Ra in Ra_values
    params = MHDParams(E=1e-3, Pr=1.0, Pm=5.0, Ra=Ra, Le=Le, ...)
    op = MHDStabilityOperator(params)
    A, B, interior_dofs, _ = assemble_mhd_matrices(op)
    result = solve_eigenvalue_problem(A[interior_dofs, interior_dofs],
                                     B[interior_dofs, interior_dofs])
    println("Ra = $Ra: σ = $(result.σ)")
end
```

### Kinematic Dynamo (No Background Field)

```julia
# Set Le = 0 for kinematic dynamo
params = MHDParams(
    E = 1e-3,
    Pr = 1.0,
    Pm = 5.0,
    Ra = 1e5,
    Le = 0.0,          # No background field
    B0_type = no_field,
    ...
)
```

---

## Benchmark Tests

### Christensen et al. (2001) Benchmark

**Case 0:** Non-magnetic convection
- E = 10⁻³, Pr = 1, Ra = 100, Pm = 5, No magnetic field
- Expected: Ra_c ≈ 50-60

**Case 1:** Strong field dynamo
- E = 10⁻³, Pr = 1, Ra = 100, Pm = 5, Le = 1
- Test Lorentz force stabilization

### Jones et al. (2011) Anelastic Benchmark

Future work: Extend to anelastic equations for more realistic planetary conditions.

---

## Physical Insights

### Dynamo Mechanisms

1. **Omega Effect (Differential Rotation)**
   - Toroidal velocity shears poloidal magnetic field
   - Creates toroidal magnetic field
   - Implemented in `operator_induction_toroidal_from_u`

2. **Alpha Effect (Helical Flows)**
   - Helical convection twists toroidal field
   - Creates poloidal magnetic field
   - Emerges from Coriolis-Lorentz interaction

3. **Magnetic Diffusion**
   - Dissipates magnetic field
   - Controlled by Em = E/Pm
   - Large Pm → slow diffusion → easier dynamo

### Stability Regimes

| Le | Ra | Regime |
|----|----|----|
| 0 | < Ra_c | Stable conduction |
| 0 | > Ra_c | Hydrodynamic convection |
| Small | > Ra_c | Weakly magnetic convection |
| O(1) | > Ra_c | Magnetoconvection |
| Large | Any | Magnetically dominated |

---

## Validation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Velocity operators | ✅ Tested | Matches Kore structure |
| Lorentz force | ⚠️ Implemented | Needs validation |
| Induction | ⚠️ Implemented | Needs validation |
| Magnetic diffusion | ⚠️ Implemented | Needs validation |
| Axial field | ⚠️ Implemented | Needs validation |
| Dipole field | ❌ Not implemented | Future work |
| Anelastic | ❌ Not implemented | Future work |

---

## References

### Papers

1. **Christensen et al. (2001)**
   "A numerical dynamo benchmark"
   *Physics of the Earth and Planetary Interiors*, 128(1-4), 25-34

2. **Jones et al. (2011)**
   "Anelastic convection-driven dynamo benchmarks"
   *Icarus*, 216(1), 120-135

3. **Dormy et al. (2004)**
   "MHD flow in a slightly differentially rotating spherical shell"
   *Earth and Planetary Science Letters*, 219(1-2), 79-86

### Codes

1. **Kore**
   - Python implementation: `kore-main/bin/operators.py`
   - Reference for operator structure

2. **PARODY**
   - Fortran dynamo code
   - Benchmark reference

3. **Magic**
   - Spectral dynamo code
   - Alternative benchmark

### Books

1. **Christensen & Wicht (2015)**
   *Numerical Dynamo Simulations*

2. **Jones (2011)**
   *Planetary Magnetic Fields and Fluid Dynamos*
   In: *Treatise on Geophysics*, Vol. 8

---

## Future Development

### High Priority

- [ ] Validate against Christensen benchmark
- [ ] Test with Jones et al. anelastic benchmark (when anelastic added)
- [ ] Implement dipole background field
- [ ] Add more magnetic field geometries

### Medium Priority

- [ ] Conducting inner core (more complex BCs)
- [ ] Variable magnetic diffusivity
- [ ] Compositional convection coupling
- [ ] Hyperdiffusion for high Ra

### Low Priority

- [ ] Torsional oscillations
- [ ] MAC waves
- [ ] Magnetic boundary layers
- [ ] Non-linear terms (DNS)

---

## Contact & Contribution

This MHD implementation follows the Kore structure and uses the corrected spectral multiplication and factorial scaling from the bug fixes (2025-10-26).

For questions or contributions, refer to the main Cross.jl documentation.

**Status:** Experimental implementation complete, validation in progress.
