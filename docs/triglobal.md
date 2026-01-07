# Tri-Global Instability Analysis

Tri-global analysis captures mode coupling across multiple azimuthal wavenumbers when the base state is fully 3-D. This enables studying instabilities driven by non-axisymmetric boundary conditions or background flows.

## When to Use Tri-Global Analysis

Use tri-global analysis when:

- **Boundary forcing varies with longitude** (e.g., hemispheric heating, topography)
- **Zonal jets** introduce azimuthal shear that couples neighboring modes
- **Magnetic fields** or compositional variations inject $m \neq 0$ components into the base state
- **Large-scale convection patterns** modify the stability of smaller-scale modes

!!! note
    If your base state is axisymmetric (`BasicState`), you can stay with single-mode onset analysis for efficiency.

## Mode Coupling Physics

### How Modes Couple

When a basic state has $m_{bs} \neq 0$ components, perturbations at mode $m$ couple to $m \pm m_{bs}$:

$$
\bar{u}_{m_{bs}} \cdot \nabla u'_m \rightarrow u'_{m + m_{bs}} + u'_{m - m_{bs}}
$$

For example, if $\bar{\Theta}$ contains $Y_{2,2}$ (so $m_{bs} = 2$):
- Perturbation mode $m=4$ couples to $m=2$ and $m=6$
- Mode $m=0$ couples to $m=2$ and $m=-2$

### Gaunt Coefficients

The coupling strength is determined by Gaunt coefficients:

$$
G_{\ell_1 \ell_2 \ell_3}^{m_1 m_2 m_3} = \int Y_{\ell_1}^{m_1} Y_{\ell_2}^{m_2} Y_{\ell_3}^{m_3*} d\Omega
$$

These are computed from Wigner 3j symbols using the `WignerSymbols.jl` package.

## Setting Up a Tri-Global Problem

### Step 1: Create a 3-D Basic State

```julia
using Cross

# Chebyshev differentiation
cd = ChebyshevDiffn(64, [0.35, 1.0], 4)

# Non-axisymmetric boundary forcing
boundary_modes = Dict(
    (2, 0) => 0.1,    # Pole-equator variation
    (2, 2) => 0.05,   # East-west variation
)

bs3d = nonaxisymmetric_basic_state(
    cd, 0.35, 1e7, 1.0;
    lmax_bs = 8,
    mmax_bs = 4,
    boundary_modes,
)
```

### Step 2: Define Tri-Global Parameters

```julia
params_triglobal = TriGlobalParams(
    # Physical parameters
    E = 1e-5,
    Pr = 1.0,
    Ra = 1.2e7,
    χ = 0.35,

    # Mode coupling range
    m_range = -2:2,           # Coupled perturbation modes

    # Resolution
    lmax = 40,
    Nr = 64,

    # Basic state
    basic_state_3d = bs3d,

    # Boundary conditions
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)
```

!!! warning "Mode Range Selection"
    `m_range` should be symmetric around your primary mode of interest to capture both forward and backward couplings.

### Step 3: Estimate Problem Size

Before solving, check the computational requirements:

```julia
size_report = estimate_triglobal_problem_size(params_triglobal)

println("Problem size estimate:")
println("  Total modes: ", size_report.total_modes)
println("  Total DOFs: ", size_report.total_dofs)
println("  Matrix size: ", size_report.matrix_size, " × ", size_report.matrix_size)
println("  Estimated memory: ", size_report.memory_estimate_gb, " GB")
```

### Typical Problem Sizes

| m_range | lmax | Nr | Approx DOFs | Memory |
|---------|------|----|-------------|--------|
| -1:1 | 30 | 32 | ~15,000 | ~2 GB |
| -2:2 | 40 | 48 | ~100,000 | ~15 GB |
| -4:4 | 50 | 64 | ~500,000 | ~100 GB |

## Setting Up and Solving

### Build the Coupled Problem

```julia
problem = setup_coupled_mode_problem(params_triglobal)

# Inspect the coupling graph
println("Coupling structure:")
for (m, neighbors) in sort(problem.coupling_graph)
    println("  Mode m=$m couples to: ", join(neighbors, ", "))
end
```

### Understand the Block Structure

The matrices have block structure where each block couples different $(m, \ell)$ pairs:

```
        m=-2   m=-1   m=0    m=1    m=2
    ┌──────────────────────────────────┐
m=-2│  A₋₂   C₋₂,₋₁  0      0      0  │
    │                                  │
m=-1│ C₋₁,₋₂  A₋₁   C₋₁,₀   0      0  │
    │                                  │
m=0 │  0     C₀,₋₁   A₀    C₀,₁    0  │
    │                                  │
m=1 │  0      0     C₁,₀    A₁    C₁,₂│
    │                                  │
m=2 │  0      0      0     C₂,₁    A₂ │
    └──────────────────────────────────┘
```

Where:
- $A_m$ = diagonal blocks (single-mode physics)
- $C_{m,m'}$ = coupling blocks from basic state interaction

### Solve the Eigenvalue Problem

```julia
solution = solve_triglobal_eigenvalue_problem(
    problem;
    nev = 12,            # Number of eigenvalues
    which = :LR,         # Largest real part
    tol = 1e-6,
)

# Leading eigenvalue
σ₁ = real(solution.values[1])
ω₁ = imag(solution.values[1])

println("Leading tri-global mode:")
println("  Growth rate: ", σ₁)
println("  Drift frequency: ", ω₁)

if σ₁ > 0
    println("  → System is UNSTABLE")
else
    println("  → System is STABLE")
end
```

## Post-Processing

### Extract Mode Components

Each eigenvector spans all coupled modes. Extract individual $m$ components:

```julia
function extract_mode(problem, eigenvector, target_m)
    idx = problem.block_indices[target_m]
    return eigenvector[idx]
end

# Get the m=0 component of the leading mode
mode0_vec = extract_mode(problem, solution.vectors[:, 1], 0)
```

### Reconstruct Physical Fields

```julia
# Reconstruct fields for each m component
fields_by_m = Dict{Int, Any}()

for m in problem.m_range
    mode_vec = extract_mode(problem, solution.vectors[:, 1], m)

    # Create temporary operator for this m
    params_m = ShellParams(
        E = params_triglobal.E,
        Pr = params_triglobal.Pr,
        Ra = params_triglobal.Ra,
        χ = params_triglobal.χ,
        m = m,
        lmax = params_triglobal.lmax,
        Nr = params_triglobal.Nr,
    )

    fields_by_m[m] = fields_from_coefficients(params_m, mode_vec; nθ=128, nφ=256)
end
```

### Analyze Mode Energy Distribution

```julia
# Compute energy in each m mode
function mode_energy(eigenvector, problem)
    energies = Dict{Int, Float64}()

    for m in problem.m_range
        mode_vec = extract_mode(problem, eigenvector, m)
        energies[m] = norm(mode_vec)^2
    end

    # Normalize
    total = sum(values(energies))
    for m in keys(energies)
        energies[m] /= total
    end

    return energies
end

energy_dist = mode_energy(solution.vectors[:, 1], problem)
println("Energy distribution:")
for m in sort(collect(keys(energy_dist)))
    println("  m = $m: ", round(100 * energy_dist[m], digits=1), "%")
end
```

## Finding Critical Parameters

### Critical Rayleigh Number Search

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

println("Critical Rayleigh number (tri-global): ", Ra_c)
```

### Parameter Sweeps

```julia
# Scan basic state amplitude
amplitudes = [0.01, 0.05, 0.1, 0.2]
results = []

for amp in amplitudes
    boundary_modes = Dict((2, 2) => amp)
    bs3d = nonaxisymmetric_basic_state(cd, χ, Ra, Pr;
        lmax_bs = 8, mmax_bs = 4, boundary_modes)

    params = TriGlobalParams(
        E = E, Pr = Pr, Ra = Ra, χ = χ,
        m_range = -2:2, lmax = 40, Nr = 64,
        basic_state_3d = bs3d,
    )

    problem = setup_coupled_mode_problem(params)
    sol = solve_triglobal_eigenvalue_problem(problem; nev=4)

    push!(results, (amplitude=amp, σ=real(sol.values[1]), ω=imag(sol.values[1])))
end
```

## Performance Tips

### Start Small

Begin with narrow `m_range` and increase gradually:

```julia
# Quick test
params_test = TriGlobalParams(..., m_range=-1:1, lmax=20, Nr=32)

# Production run
params_full = TriGlobalParams(..., m_range=-3:3, lmax=50, Nr=64)
```

### Use Sparse Storage

Keep basic state dictionaries sparse - only populate non-zero modes:

```julia
# Good: Only include active modes
boundary_modes = Dict((2, 2) => 0.1)

# Avoid: Don't fill with zeros
# boundary_modes = Dict((l, m) => 0.0 for l in 0:10, m in -l:l)
```

### Solver Fallback

If the default solver stalls, try KrylovKit:

```julia
solution = solve_triglobal_eigenvalue_problem(
    problem;
    nev = 12,
    solver = :krylov,    # Alternative solver
    tol = 1e-5,
    maxiter = 500,
)
```

## Complete Example

```julia
#!/usr/bin/env julia
# triglobal_analysis.jl

using Cross
using Printf

# === Parameters ===
E = 1e-5
Pr = 1.0
Ra = 1.5e7
χ = 0.35
Nr = 48

# === Basic State ===
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

boundary_modes = Dict(
    (2, 0) => 0.1,
    (2, 2) => 0.08,
)

bs3d = nonaxisymmetric_basic_state(cd, χ, Ra, Pr;
    lmax_bs = 8, mmax_bs = 4, boundary_modes)

# === Tri-Global Setup ===
params = TriGlobalParams(
    E = E, Pr = Pr, Ra = Ra, χ = χ,
    m_range = -2:2,
    lmax = 35,
    Nr = Nr,
    basic_state_3d = bs3d,
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)

# === Check Size ===
size_report = estimate_triglobal_problem_size(params)
@printf("Problem: %d DOFs, ~%.1f GB\n",
    size_report.total_dofs, size_report.memory_estimate_gb)

# === Solve ===
println("Building coupled problem...")
problem = setup_coupled_mode_problem(params)

println("Solving eigenvalue problem...")
solution = solve_triglobal_eigenvalue_problem(problem; nev=8)

# === Results ===
println("\n" * "="^50)
println("Leading eigenvalues:")
for (i, λ) in enumerate(solution.values[1:min(5, length(solution.values))])
    @printf("  %d: σ = %+.6e, ω = %+.6f\n", i, real(λ), imag(λ))
end

if real(solution.values[1]) > 0
    println("\nSystem is UNSTABLE")
else
    println("\nSystem is STABLE")
end
```

## Checklist

Before running tri-global analysis:

- [ ] `TriGlobalParams` uses m-range consistent with basic state content
- [ ] Coupling graph matches physical expectations
- [ ] Estimated problem size is feasible for available memory
- [ ] Basic state satisfies reality conditions
- [ ] Solver converges within reasonable iteration limit

## Next Steps

- **[MHD Extension](mhd_extension.md)** - Add magnetic field effects
- **[API Reference](reference.md)** - Complete function documentation

---

!!! info "Example Scripts"
    See `example/triglobal_analysis_demo.jl` for a complete working example.
