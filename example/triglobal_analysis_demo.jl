#!/usr/bin/env julia
#
# Tri-Global Instability Analysis - Framework Demonstration
#
# This example demonstrates the framework for tri-global analysis where
# perturbations couple across multiple azimuthal modes m due to a
# non-axisymmetric basic state.
#
# NOTE: The full eigenvalue solver is not yet implemented. This example
# shows how to set up the problem and analyze the coupling structure.

using Cross
using Printf

println("="^70)
println("Tri-Global Instability Analysis Framework")
println("="^70)
println()

# =============================================================================
# Step 1: Create a 3D Basic State
# =============================================================================
println("STEP 1: Create Non-Axisymmetric Basic State")
println("-"^70)
println()

# Physical parameters
E = 1e-5
Pr = 1.0
Ra = 1e7
χ = 0.35

# Basic state parameters
lmax_bs = 4
mmax_bs = 2  # Key: Non-zero m modes create mode coupling!
Nr = 64

println("Creating 3D basic state with m_bs = $mmax_bs...")
println()

# Create Chebyshev grid
cd = ChebyshevDiffn(Nr, [χ, 1.0], 2)

# Define 3D boundary temperature pattern
amplitudes = Dict(
    (2, 0) => 0.10,   # Axisymmetric: pole-to-equator
    (2, 2) => 0.05    # Non-axisymmetric: wavenumber-2 pattern
)

println("Boundary temperature modes:")
for ((ℓ, m), amp) in sort(collect(amplitudes))
    println("  Y_$(ℓ)$(m): amplitude = ", @sprintf("%.3f", amp))
end
println()

# Create basic state
bs3d = nonaxisymmetric_basic_state(cd, χ, Ra, Pr, lmax_bs, mmax_bs, amplitudes)

println("✓ Basic state created")
println("  Contains modes: ")
for ((ℓ, m), theta) in sort(collect(bs3d.theta_coeffs))
    if maximum(abs.(theta)) > 1e-10
        println("    (ℓ,m) = ($ℓ,$m)")
    end
end
println()

# =============================================================================
# Step 2: Analyze Mode Coupling Structure
# =============================================================================
println("="^70)
println("STEP 2: Analyze Mode Coupling Structure")
println("-"^70)
println()

# Find non-zero azimuthal modes in basic state
m_bs_modes = Int[]
for ((ℓ, m_bs), theta) in bs3d.theta_coeffs
    if m_bs != 0 && maximum(abs.(theta)) > 1e-10
        push!(m_bs_modes, m_bs)
    end
end
m_bs_modes = sort(unique(m_bs_modes))

println("Non-zero azimuthal modes in basic state: ", m_bs_modes)
println()

# Define perturbation mode range
# With m_bs = 2, we need to include modes that couple: m, m±2
m_range = -4:4

println("Perturbation mode range: ", m_range)
println("Number of perturbation modes: ", length(m_range))
println()

# Analyze coupling for each perturbation mode
println("Mode Coupling Structure:")
println(@sprintf("%-15s %-30s %-s", "Pert. Mode", "Couples To", "Explanation"))
println("-"^70)

for m in m_range
    # Determine coupled modes
    coupled = Int[m]  # Always couples to itself

    # Add coupling through each basic state mode
    for m_bs in m_bs_modes
        if (m - m_bs) in m_range && (m - m_bs) != m
            push!(coupled, m - m_bs)
        end
        if (m + m_bs) in m_range && (m + m_bs) != m
            push!(coupled, m + m_bs)
        end
    end

    coupled = sort(unique(coupled))

    # Explanation
    if length(coupled) == 1
        explanation = "Isolated (no basic state forcing)"
    else
        explanation = "Coupled via m_bs=$(m_bs_modes)"
    end

    println(@sprintf("m = %-12d %-30s %s", m, string(coupled), explanation))
end
println()

# =============================================================================
# Step 3: Set Up Tri-Global Problem
# =============================================================================
println("="^70)
println("STEP 3: Set Up Tri-Global Eigenvalue Problem")
println("-"^70)
println()

# Create tri-global parameters
lmax = 50  # Max ℓ for perturbations

params_triglobal = TriglobalParams(
    E = E,
    Pr = Pr,
    Ra = Ra,
    χ = χ,
    m_range = m_range,
    lmax = lmax,
    Nr = Nr,
    basic_state_3d = bs3d,
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature
)

println("Tri-global parameters:")
println("  E      = ", E)
println("  Pr     = ", Pr)
println("  Ra     = ", Ra)
println("  χ      = ", χ)
println("  m_range= ", m_range)
println("  lmax   = ", lmax)
println("  Nr     = ", Nr)
println()

# Set up coupled problem structure
problem = setup_coupled_mode_problem(params_triglobal)

println("✓ Problem structure created")
println()
println("Problem Statistics:")
println("  Perturbation modes: ", problem.m_range)
println("  Basic state modes:  ", problem.all_m_bs)
println("  Total DOFs:         ", problem.total_dofs)
println()

# Estimate problem size
size_info = estimate_triglobal_problem_size(params_triglobal)

println("Problem Size Estimate:")
println("  Total degrees of freedom: ", size_info.total_dofs)
println("  Matrix size:              ", size_info.matrix_size, " × ", size_info.matrix_size)
println("  Number of coupled modes:  ", size_info.num_modes)
println("  Average DOFs per mode:    ", size_info.dofs_per_mode)
println()

# Memory estimate
memory_gb = (size_info.matrix_size^2 * 8) / 1e9  # 8 bytes per Float64
println("  Estimated memory (dense): ", @sprintf("%.2f GB", memory_gb))
println("  Note: Use SPARSE matrices for practical computation!")
println()

# =============================================================================
# Step 4: Display Coupling Graph
# =============================================================================
println("="^70)
println("STEP 4: Mode Coupling Graph")
println("-"^70)
println()

println("Coupling structure (perturbation modes):")
for m in problem.m_range
    coupled_to = problem.coupling_graph[m]
    block_range = problem.block_indices[m]

    println("  m = ", @sprintf("%2d", m), ": couples to ",
            @sprintf("%-20s", string(coupled_to)),
            " | DOF range: ", block_range.start, ":", block_range.stop,
            " (", length(block_range), " DOFs)")
end
println()

# =============================================================================
# Step 5: Comparison with Standard Onset
# =============================================================================
println("="^70)
println("STEP 5: Comparison with Standard (Single-m) Onset")
println("-"^70)
println()

println("Standard Onset (single m):")
println("  - Solves for ONE azimuthal mode m at a time")
println("  - Problem size: O(lmax × Nr × 3) ≈ ", lmax * Nr * 3, " DOFs")
println("  - Eigenvalue problem: A_m x = λ B_m x")
println("  - Computational cost: Moderate")
println()

println("Tri-Global Onset (coupled modes):")
println("  - Solves for MULTIPLE modes m simultaneously")
println("  - Problem size: O(|m_range| × lmax × Nr × 3) ≈ ", size_info.total_dofs, " DOFs")
println("  - Eigenvalue problem: BLOCK-COUPLED across m")
println("  - Computational cost: ", size_info.num_modes, "× larger")
println()

println("When is tri-global analysis necessary?")
println("  ✓ Basic state has non-axisymmetric components (m_bs ≠ 0)")
println("  ✓ Studying effects of longitudinal variations")
println("  ✓ Realistic 3D boundary conditions")
println("  ✓ Mode interactions are important")
println()

println("When can you use standard (single-m) analysis?")
println("  ✓ Basic state is axisymmetric (m_bs = 0 only)")
println("  ✓ Modes decouple: no longitudinal variations")
println("  ✓ Much faster and more practical for large problems")
println()

# =============================================================================
# Summary and Next Steps
# =============================================================================
println("="^70)
println("SUMMARY AND NEXT STEPS")
println("="^70)
println()

println("✓ COMPLETED:")
println("  1. Created 3D non-axisymmetric basic state")
println("  2. Analyzed mode coupling structure")
println("  3. Set up tri-global problem framework")
println("  4. Estimated computational requirements")
println()

println("⚠ TODO (Future Implementation):")
println("  1. Build single-mode operators for each m")
println("  2. Implement mode-coupling operators:")
println("     - Advection: (ū_bs · ∇)θ' couples different m values")
println("     - Advection: (u' · ∇)θ̄_bs couples different m values")
println("  3. Assemble block-sparse coupled matrix")
println("  4. Solve large eigenvalue problem using:")
println("     - Shift-invert with Krylov methods")
println("     - Sparse matrix storage")
println("     - Iterative linear solvers (GMRES, BiCGStab)")
println("  5. Find critical Rayleigh number via root-finding")
println()

println("Physical Questions to Address:")
println("  • How do longitudinal temperature variations affect Ra_c?")
println("  • Does wavenumber-2 boundary heating prefer m=2 perturbations?")
println("  • How strong is the mode coupling for small amplitudes?")
println("  • Can we use perturbation theory for weak coupling?")
println()

println("Computational Strategy:")
println("  • Start with small problems (few coupled modes)")
println("  • Use iterative eigensolvers (not full diagonalization!)")
println("  • Exploit sparsity in coupling (most modes don't couple)")
println("  • Consider reduced models for parameter studies")
println()

println("="^70)
println()
println("Framework is ready for implementation of coupled eigenvalue solver!")
println()
