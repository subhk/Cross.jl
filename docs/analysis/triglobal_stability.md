# Triglobal Stability Analysis with Non-Axisymmetric Mean Flow

Triglobal stability analysis handles the most general case: fully three-dimensional basic states with non-axisymmetric ($m \neq 0$) components. This introduces **mode coupling** between perturbations at different azimuthal wavenumbers, requiring simultaneous solution of coupled modes.

## Physical Motivation

### When Triglobal Analysis is Required

Triglobal analysis is necessary when the background state breaks axisymmetry:

1. **Hemispheric asymmetry** - Different heat flux between hemispheres
2. **Topographic forcing** - Non-axisymmetric core-mantle boundary
3. **Large-scale convection** - Pre-existing convective patterns modifying stability
4. **Magnetic field effects** - Non-axisymmetric imposed fields
5. **Tidal forcing** - Periodic longitudinal variations
6. **Laboratory experiments** - Asymmetric heating or boundary conditions

### Real-World Applications

| System | Source of Non-Axisymmetry |
|--------|--------------------------|
| Earth's core | CMB heat flux heterogeneity, inner core asymmetry |
| Mercury | 3:2 spin-orbit resonance |
| Io, Europa | Tidal heating patterns |
| Giant planets | Non-axisymmetric deep jets |
| Stars | Active regions, spot coverage |

## Mode Coupling Physics

### The Coupling Mechanism

When the basic state contains $m_{bs} \neq 0$ components, the advection terms couple different perturbation modes:

$$
\bar{u}_{m_{bs}} \cdot \nabla u'_m \rightarrow u'_{m + m_{bs}} + u'_{m - m_{bs}}
$$

**Example**: If $\bar{\Theta}$ contains $Y_2^2$ (so $m_{bs} = 2$):
- Perturbation at $m = 4$ couples to $m = 6$ and $m = 2$
- Perturbation at $m = 0$ couples to $m = 2$ and $m = -2$
- A cascade of couplings connects all modes differing by multiples of $m_{bs}$

### Gaunt Coefficients

The coupling strength is determined by **Gaunt coefficients**:

$$
G_{\ell_1 \ell_2 \ell_3}^{m_1 m_2 m_3} = \int Y_{\ell_1}^{m_1} Y_{\ell_2}^{m_2} Y_{\ell_3}^{m_3*} \, d\Omega
$$

These are computed from **Wigner 3j symbols**:

$$
G_{\ell_1 \ell_2 \ell_3}^{m_1 m_2 m_3} = \sqrt{\frac{(2\ell_1+1)(2\ell_2+1)(2\ell_3+1)}{4\pi}}
\begin{pmatrix} \ell_1 & \ell_2 & \ell_3 \\ 0 & 0 & 0 \end{pmatrix}
\begin{pmatrix} \ell_1 & \ell_2 & \ell_3 \\ m_1 & m_2 & m_3 \end{pmatrix}
$$

Selection rules:
- $m_1 + m_2 = m_3$
- $|\ell_1 - \ell_2| \leq \ell_3 \leq \ell_1 + \ell_2$ (triangle inequality)
- $\ell_1 + \ell_2 + \ell_3$ must be even

### Block Matrix Structure

The coupled eigenvalue problem has block structure:

```
        m=-2   m=-1   m=0    m=1    m=2
    ┌──────────────────────────────────┐
m=-2│  A₋₂   C₋₂,₋₁  0      0      0   │
    │                                  │
m=-1│ C₋₁,₋₂  A₋₁   C₋₁,₀   0      0   │
    │                                  │
m=0 │  0     C₀,₋₁   A₀    C₀,₁    0   │
    │                                  │
m=1 │  0      0     C₁,₀    A₁    C₁,₂ │
    │                                  │
m=2 │  0      0      0     C₂,₁    A₂  │
    └──────────────────────────────────┘
```

Where:
- $A_m$ = diagonal blocks (single-mode physics: diffusion, Coriolis, buoyancy)
- $C_{m,m'}$ = coupling blocks from basic state advection

## Mathematical Formulation

### Full 3D Basic State

The basic state contains all spherical harmonic components:

**Temperature:**
$$
\bar{T}(r, \theta, \phi) = \sum_{\ell=0}^{L_{bs}} \sum_{m=-\ell}^{\ell} \bar{\Theta}_{\ell m}(r) Y_\ell^m(\theta, \phi)
$$

**Velocity:**
$$
\bar{\mathbf{u}}(r, \theta, \phi) = \sum_{\ell, m} \left[ \bar{u}_{r,\ell m}(r) Y_\ell^m \hat{\mathbf{r}} + \bar{u}_{\theta,\ell m}(r) \nabla_H Y_\ell^m + \bar{u}_{\phi,\ell m}(r) \hat{\mathbf{r}} \times \nabla_H Y_\ell^m \right]
$$

### Coupled Perturbation Equations

For perturbations spanning $m \in [m_{min}, m_{max}]$:

$$
\frac{\partial \mathbf{u}'_m}{\partial t} + 2\hat{\mathbf{z}} \times \mathbf{u}'_m + \sum_{m'} \left[ (\mathbf{u}'_m \cdot \nabla)\bar{\mathbf{u}}_{m-m'} + (\bar{\mathbf{u}}_{m'} \cdot \nabla)\mathbf{u}'_{m-m'} \right] = \ldots
$$

The sum couples modes $m$ and $m - m'$ through basic state component $m'$.

### Generalized Eigenvalue Problem

$$
\begin{pmatrix}
\mathbf{A}_{-2} & \mathbf{C}_{-2,-1} & & & \\
\mathbf{C}_{-1,-2} & \mathbf{A}_{-1} & \mathbf{C}_{-1,0} & & \\
& \mathbf{C}_{0,-1} & \mathbf{A}_0 & \mathbf{C}_{0,1} & \\
& & \mathbf{C}_{1,0} & \mathbf{A}_1 & \mathbf{C}_{1,2} \\
& & & \mathbf{C}_{2,1} & \mathbf{A}_2
\end{pmatrix}
\begin{pmatrix}
\mathbf{x}_{-2} \\ \mathbf{x}_{-1} \\ \mathbf{x}_0 \\ \mathbf{x}_1 \\ \mathbf{x}_2
\end{pmatrix}
= \sigma
\begin{pmatrix}
\mathbf{B}_{-2} & & & & \\
& \mathbf{B}_{-1} & & & \\
& & \mathbf{B}_0 & & \\
& & & \mathbf{B}_1 & \\
& & & & \mathbf{B}_2
\end{pmatrix}
\begin{pmatrix}
\mathbf{x}_{-2} \\ \mathbf{x}_{-1} \\ \mathbf{x}_0 \\ \mathbf{x}_1 \\ \mathbf{x}_2
\end{pmatrix}
$$

## The `BasicState3D` Structure

```julia
struct BasicState3D{T}
    lmax_bs::Int
    mmax_bs::Int
    Nr::Int
    r::Vector{T}

    # Temperature: θ̄_ℓm(r) indexed by (ℓ, m)
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    dtheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}

    # Velocity components: ū_r,ℓm(r), ū_θ,ℓm(r), ū_φ,ℓm(r)
    ur_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    utheta_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}}

    # Velocity derivatives
    dur_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    dutheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    duphi_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
end
```

### Reality Conditions

For physical (real-valued) fields, coefficients must satisfy:

$$
\bar{f}_{\ell,-m} = (-1)^m \bar{f}_{\ell,m}^*
$$

This is automatically enforced when constructing `BasicState3D` from physical data.

## Creating 3D Basic States

### Method 1: From Boundary Conditions

Solve $\nabla^2 \bar{T} = 0$ with non-axisymmetric boundary heating:

```julia
using Cross

# Chebyshev setup
Nr = 64
χ = 0.35
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# Define non-axisymmetric boundary modes
boundary_modes = Dict(
    (2, 0) => 0.10,   # Y₂₀: pole-equator variation
    (2, 2) => 0.05,   # Y₂₂: east-west variation
    (3, 2) => 0.02,   # Y₃₂: higher order
)

# Create 3D basic state
bs3d = nonaxisymmetric_basic_state(
    cd, χ, Ra, Pr;
    lmax_bs = 8,
    mmax_bs = 4,
    boundary_modes = boundary_modes,
)
```

### Method 2: Manual Construction

For custom profiles from simulations:

```julia
# Initialize dictionaries
Nr = 64
lmax_bs = 8
mmax_bs = 3
r = cd.x

theta_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}()
dtheta_dr_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}()
# ... other coefficient dictionaries ...

# Populate for all (ℓ, m) pairs
for ℓ in 0:lmax_bs
    for m in -min(ℓ, mmax_bs):min(ℓ, mmax_bs)
        theta_coeffs[(ℓ, m)] = zeros(Nr)
        dtheta_dr_coeffs[(ℓ, m)] = zeros(Nr)
    end
end

# Set specific mode amplitudes
theta_coeffs[(2, 0)] .= your_T20_profile
theta_coeffs[(2, 2)] .= your_T22_profile

# Enforce reality condition
theta_coeffs[(2, -2)] .= conj.(theta_coeffs[(2, 2)])

# Compute derivatives
for (ℓm, coeffs) in theta_coeffs
    dtheta_dr_coeffs[ℓm] = cd.D1 * coeffs
end

# Construct BasicState3D
bs3d = BasicState3D(
    r = r, Nr = Nr,
    lmax_bs = lmax_bs, mmax_bs = mmax_bs,
    theta_coeffs = theta_coeffs,
    dtheta_dr_coeffs = dtheta_dr_coeffs,
    # ... velocity coefficients ...
)
```

### Method 3: Import from Simulation

```julia
using JLD2
using Interpolations

# Load spectral coefficients from external code
@load "simulation_3d.jld2" T_lm u_lm r_sim

# Interpolate to Cross.jl grid
for (ℓ, m) in keys(T_lm)
    itp = LinearInterpolation(r_sim, T_lm[(ℓ, m)])
    theta_coeffs[(ℓ, m)] = itp.(cd.x)
    dtheta_dr_coeffs[(ℓ, m)] = cd.D1 * theta_coeffs[(ℓ, m)]
end
```

## Triglobal Analysis Workflow

### Step 1: Create 3D Basic State

```julia
using Cross

# Parameters
E = 1e-5
Pr = 1.0
Ra = 1.5e7
χ = 0.35
Nr = 48

# Chebyshev operators
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# Non-axisymmetric boundary forcing
boundary_modes = Dict(
    (2, 0) => 0.10,   # Axisymmetric part
    (2, 2) => 0.08,   # Non-axisymmetric: m = 2
)

bs3d = nonaxisymmetric_basic_state(cd, χ, Ra, Pr;
    lmax_bs = 8, mmax_bs = 4, boundary_modes)
```

### Step 2: Define Triglobal Parameters

```julia
params_triglobal = TriglobalParams(
    # Physical parameters
    E = E,
    Pr = Pr,
    Ra = Ra,
    χ = χ,

    # Mode coupling range
    m_range = -2:2,           # Coupled perturbation modes

    # Resolution
    lmax = 40,
    Nr = Nr,

    # 3D Basic state
    basic_state_3d = bs3d,

    # Boundary conditions
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)
```

!!! warning "Mode Range Selection"
    Choose `m_range` to be symmetric and wide enough to capture the coupling cascade. If the basic state has $m_{bs} = 2$, modes separated by 2 will couple.

### Step 3: Estimate Problem Size

Before solving, check computational requirements:

```julia
size_report = estimate_triglobal_problem_size(params_triglobal)

println("Triglobal Problem Size:")
println("  Number of modes:     ", size_report.num_modes)
println("  Total DOFs:          ", size_report.total_dofs)
println("  Matrix dimensions:   ", size_report.matrix_size, " × ", size_report.matrix_size)
println("  DOFs per mode:       ", size_report.dofs_per_mode)
```

### Typical Problem Sizes

| m_range | lmax | Nr | Approx DOFs |
|---------|------|----|-------------|
| -1:1 | 30 | 32 | ~15,000 |
| -2:2 | 40 | 48 | ~100,000 |
| -3:3 | 45 | 56 | ~250,000 |
| -4:4 | 50 | 64 | ~500,000 |

### Step 4: Build and Solve

```julia
# Build coupled problem
println("Building coupled-mode problem...")
problem = setup_coupled_mode_problem(params_triglobal)

# Inspect coupling structure
println("Coupling graph:")
for (m, neighbors) in sort(problem.coupling_graph)
    println("  m = $m couples to: ", join(neighbors, ", "))
end

# Solve eigenvalue problem
println("Solving eigenvalue problem...")
eigenvalues, eigenvectors = solve_triglobal_eigenvalue_problem(
    params_triglobal;
    nev = 12,            # Number of eigenvalues
    σ_target = 0.0,
    verbose = true,
)

# Results
σ₁ = real(eigenvalues[1])
ω₁ = imag(eigenvalues[1])

println("\nLeading triglobal mode:")
println("  Growth rate: σ = $σ₁")
println("  Drift frequency: ω = $ω₁")
println("  Status: ", σ₁ > 0 ? "UNSTABLE" : "STABLE")
```

## Post-Processing

### Extract Mode Components

Each eigenvector spans all coupled $m$ values:

```julia
function extract_mode_component(problem, eigenvector, target_m)
    idx = problem.block_indices[target_m]
    return eigenvector[idx]
end

# Get m=0 component of leading mode
mode0_coeffs = extract_mode_component(problem, eigenvectors[:, 1], 0)

# Get m=2 component
mode2_coeffs = extract_mode_component(problem, eigenvectors[:, 1], 2)
```

### Analyze Mode Energy Distribution

```julia
function mode_energy_distribution(eigenvector, problem)
    energies = Dict{Int, Float64}()

    for m in problem.m_range
        mode_vec = extract_mode_component(problem, eigenvector, m)
        energies[m] = norm(mode_vec)^2
    end

    # Normalize to percentages
    total = sum(values(energies))
    for m in keys(energies)
        energies[m] = 100.0 * energies[m] / total
    end

    return energies
end

# Compute energy distribution
energy_dist = mode_energy_distribution(eigenvectors[:, 1], problem)

println("Energy distribution in leading mode:")
for m in sort(collect(keys(energy_dist)))
    @printf("  m = %+2d: %.1f%%\n", m, energy_dist[m])
end
```

### Reconstruct Physical Fields

```julia
# Each block contains interior DOFs for a single m.
for m in problem.m_range
    mode_vec = extract_mode_component(problem, eigenvectors[:, 1], m)
    # mode_vec contains interior DOFs for this m block.
end
```

## Finding Critical Parameters

### Critical Rayleigh Number

```julia
Ra_c, ω_c, eigvec = find_critical_rayleigh_triglobal(
    E = params_triglobal.E,
    Pr = params_triglobal.Pr,
    χ = params_triglobal.χ,
    m_range = params_triglobal.m_range,
    lmax = params_triglobal.lmax,
    Nr = params_triglobal.Nr,
    basic_state_3d = bs3d;
    Ra_guess = 1e7,
    tol = 1e-3,
)

println("Triglobal critical Rayleigh: Ra_c = $Ra_c")
```

### Parameter Sweeps

```julia
# Sweep basic state amplitude
amplitudes = [0.0, 0.02, 0.05, 0.1, 0.2]
results_sweep = []

for amp in amplitudes
    @printf("Amplitude = %.2f: ", amp)

    if amp == 0.0
        # Axisymmetric only
        boundary_modes = Dict((2, 0) => 0.1)
    else
        # Add non-axisymmetric component
        boundary_modes = Dict(
            (2, 0) => 0.1,
            (2, 2) => amp,
        )
    end

    bs3d = nonaxisymmetric_basic_state(cd, χ, Ra, Pr;
        lmax_bs = 8, mmax_bs = 4, boundary_modes)

    params = TriglobalParams(
        E = E, Pr = Pr, Ra = Ra, χ = χ,
        m_range = -2:2, lmax = 40, Nr = Nr,
        basic_state_3d = bs3d,
    )

    eigenvalues, _ = solve_triglobal_eigenvalue_problem(params; nev=4, verbose=false)

    σ = real(eigenvalues[1])
    ω = imag(eigenvalues[1])

    push!(results_sweep, (amplitude=amp, σ=σ, ω=ω))
    @printf("σ = %+.4e, ω = %+.4f\n", σ, ω)
end
```

## Performance Optimization

### Start Small

Begin with narrow mode range and increase:

```julia
# Quick test run
params_test = TriglobalParams(...,
    m_range = -1:1,
    lmax = 25,
    Nr = 32,
)

# Verify before production run
params_full = TriglobalParams(...,
    m_range = -3:3,
    lmax = 50,
    Nr = 64,
)
```

### Sparse Basic State Storage

Only populate non-zero modes:

```julia
# Good: sparse storage
boundary_modes = Dict((2, 2) => 0.1)  # Only non-zero modes

# Bad: dense storage (unnecessary)
# boundary_modes = Dict((ℓ, m) => 0.0 for ℓ in 0:10, m in -ℓ:ℓ)
```

## Complete Example

```julia
#!/usr/bin/env julia
# triglobal_complete_analysis.jl
#
# Triglobal stability analysis with non-axisymmetric basic state

using Cross
using Printf
using JLD2

# === Parameters ===
E = 1e-5
Pr = 1.0
Ra = 1.5e7
χ = 0.35
Nr = 48

# === Setup ===
println("="^60)
println("Triglobal Stability Analysis")
println("Non-Axisymmetric Basic State with Mode Coupling")
println("="^60)

cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# === Create 3D Basic State ===
boundary_modes = Dict(
    (2, 0) => 0.10,   # Pole-equator (axisymmetric)
    (2, 2) => 0.08,   # East-west (non-axisymmetric)
)

println("\nBasic state modes:")
for ((ℓ, m), amp) in boundary_modes
    println("  Y($ℓ,$m) amplitude = $amp")
end

bs3d = nonaxisymmetric_basic_state(cd, χ, Ra, Pr;
    lmax_bs = 8, mmax_bs = 4, boundary_modes)

# === Triglobal Setup ===
params = TriglobalParams(
    E = E, Pr = Pr, Ra = Ra, χ = χ,
    m_range = -2:2,
    lmax = 35,
    Nr = Nr,
    basic_state_3d = bs3d,
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)

# === Problem Size ===
size_report = estimate_triglobal_problem_size(params)
@printf("\nProblem size: %d DOFs across %d modes\n",
    size_report.total_dofs, size_report.num_modes)

# === Build & Solve ===
println("\nBuilding coupled problem...")
problem = setup_coupled_mode_problem(params)

println("Coupling structure:")
for (m, neighbors) in sort(problem.coupling_graph)
    println("  m = $m ↔ ", join(neighbors, ", "))
end

println("\nSolving eigenvalue problem...")
eigenvalues, eigenvectors = solve_triglobal_eigenvalue_problem(params; nev=8)

# === Results ===
println("\n" * "="^60)
println("RESULTS")
println("="^60)

println("\nLeading eigenvalues:")
for (i, λ) in enumerate(eigenvalues[1:min(5, length(eigenvalues))])
    @printf("  %d: σ = %+.6e, ω = %+.6f\n", i, real(λ), imag(λ))
end

σ₁ = real(eigenvalues[1])
status = σ₁ > 0 ? "UNSTABLE" : "STABLE"
println("\nSystem is $status at Ra = $Ra")

# === Energy Distribution ===
println("\nEnergy distribution (leading mode):")
total_E = 0.0
mode_E = Dict{Int, Float64}()

for m in params.m_range
    idx = problem.block_indices[m]
    E_m = norm(eigenvectors[idx, 1])^2
    mode_E[m] = E_m
    total_E += E_m
end

for m in sort(collect(keys(mode_E)))
    pct = 100.0 * mode_E[m] / total_E
    @printf("  m = %+2d: %.1f%%\n", m, pct)
end

# === Comparison to Biglobal ===
println("\n" * "="^60)
println("COMPARISON: Triglobal vs Biglobal")
println("="^60)

# Biglobal (axisymmetric basic state only)
bs_axi = meridional_basic_state(cd, χ, E, Ra, Pr;
    lmax_bs = 6, amplitude = 0.1)

params_bi = ShellParams(
    E = E, Pr = Pr, Ra = Ra, χ = χ,
    m = 0, lmax = 35, Nr = Nr,
    basic_state = bs_axi,
)

eigenvalues_bi, _, _, _ = leading_modes(params_bi; nev=4)
σ_biglobal = real(eigenvalues_bi[1])

@printf("\n  Biglobal (m=0 only):  σ = %+.6e\n", σ_biglobal)
@printf("  Triglobal (coupled):  σ = %+.6e\n", σ₁)
@printf("  Difference:           Δσ = %+.6e\n", σ₁ - σ_biglobal)

# === Save Results ===
@save "outputs/triglobal_analysis.jld2" eigenvalues eigenvectors params E Pr Ra χ boundary_modes

println("\nResults saved to outputs/triglobal_analysis.jld2")
```

## Checklist

Before running triglobal analysis:

- [ ] `m_range` is symmetric (e.g., -2:2, not 0:4)
- [ ] `m_range` covers coupling from basic state $m_{bs}$
- [ ] Problem size fits in available memory
- [ ] Basic state satisfies reality conditions
- [ ] Basic state `mmax_bs` matches expected coupling
- [ ] Solver converges within iteration limit
- [ ] Energy distribution shows expected mode participation

## Comparison of Analysis Types

| Feature | Onset (No Flow) | Biglobal (Axisymm.) | Triglobal (3D) |
|---------|-----------------|---------------------|----------------|
| Basic state $m$ | 0 only | 0 only | All $m$ |
| Mode coupling | None | None | Yes |
| Matrix structure | Block diagonal | Block diagonal | Coupled blocks |
| DOFs per $m$ | $N_r \times N_\ell$ | $N_r \times N_\ell$ | $N_r \times N_\ell$ |
| Total DOFs | Single $m$ | Single $m$ | $\sum_m N_r \times N_\ell$ |
| Memory | Low | Low | High |
| Applications | Classical onset | Thermal wind | CMB heterogeneity |

## Next Steps

- **[Onset Convection](onset_convection.md)** - Classical problem without mean flow
- **[Biglobal Stability](biglobal_stability.md)** - Axisymmetric mean flows

---

!!! info "Example Scripts"
    See `example/triglobal_analysis_demo.jl` and `example/nonaxisymmetric_basic_state.jl` for complete working examples.
