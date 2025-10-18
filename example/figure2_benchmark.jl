#!/usr/bin/env julia
#
# Benchmark script reproducing the neutral-stability curve shown in
# Figure 2 of docs/Onset_convection.pdf (Barik et al., 2023).
#
# This script scans azimuthal wavenumbers m from 5 to 30 and computes
# the critical Rayleigh number Ra_c where the growth rate σ = 0 for
# each m. The parameters match Figure 2:
#   E = 10^-5, χ = 0.35, Pr = 1.0
#
# Expected results from the paper:
#   Critical point: m_c = 15, Ra_c ≈ 1.05567 × 10^7

using Cross
using Printf

# Physical parameters matching Figure 2 of Barik et al. (2023)
const E = 1e-5      # Ekman number
const Pr = 1.0      # Prandtl number
const χ = 0.35      # Radius ratio r_i/r_o

# Numerical resolution
const lmax = 30     # Maximum spherical harmonic degree
const Nr = 32       # Number of radial collocation points

# Expected values from paper (for comparison)
const m_c_expected = 15
const Ra_c_expected = 1.05567e7

println("="^70)
println("Reproducing Figure 2 from Barik et al. (2023)")
println("Onset of convection in rotating spherical shell")
println("="^70)
println()
println("Parameters:")
println("  E  = ", E)
println("  Pr = ", Pr)
println("  χ  = ", χ)
println("  lmax = ", lmax)
println("  Nr = ", Nr)
println()
println("Expected critical point from paper:")
println("  m_c  = ", m_c_expected)
println("  Ra_c = ", Ra_c_expected)
println()
println("="^70)
println()

# Storage for results
m_values = Int[]
Ra_critical = Float64[]
ω_critical = Float64[]

println(@sprintf("%-5s %-15s %-15s %-10s", "m", "Ra_c", "ω_c", "Status"))
println("-"^70)

# Scan azimuthal wavenumbers
for m in 1:30
    try
        # Initial guess based on expected scaling
        Ra_guess = 2e7

        # Find critical Rayleigh number for this m
        Ra_c, ω_c, vec = find_critical_rayleigh(
            E, Pr, χ, m, lmax, Nr;
            Ra_guess = Ra_guess,
            Ra_bracket = (Ra_guess * 0.3, Ra_guess * 3.0),
            mechanical_bc = :no_slip,
            thermal_bc = :fixed_temperature,
            tol = 1e-6
        )

        push!(m_values, m)
        push!(Ra_critical, Ra_c)
        push!(ω_critical, ω_c)

        println(@sprintf("%-5d %-15.6e %-15.6f %-10s", m, Ra_c, ω_c, "OK"))

    catch e
        println(@sprintf("%-5d %-15s %-15s %-10s", m, "FAILED", "-", "ERROR"))
        @warn "Failed for m = $m" exception=e
    end
end

println()
println("="^70)
println("Results Summary")
println("="^70)
println()

if !isempty(Ra_critical)
    # Find the minimum (critical point)
    idx_min = argmin(Ra_critical)
    m_c = m_values[idx_min]
    Ra_c_min = Ra_critical[idx_min]
    ω_c_min = ω_critical[idx_min]

    println("Critical point found:")
    println("  m_c  = ", m_c)
    println("  Ra_c = ", Ra_c_min)
    println("  ω_c  = ", ω_c_min)
    println()

    # Compare with expected values
    pct_diff_m = 100 * abs(m_c - m_c_expected) / m_c_expected
    pct_diff_Ra = 100 * abs(Ra_c_min - Ra_c_expected) / Ra_c_expected

    println("Comparison with Barik et al. (2023):")
    println("  m_c:  computed = ", m_c, ", expected = ", m_c_expected,
            " (", @sprintf("%.2f", pct_diff_m), "% difference)")
    println("  Ra_c: computed = ", @sprintf("%.5e", Ra_c_min),
            ", expected = ", @sprintf("%.5e", Ra_c_expected),
            " (", @sprintf("%.2f", pct_diff_Ra), "% difference)")
    println()

    if pct_diff_Ra < 1.0
        println("Excellent agreement with published results!")
    elseif pct_diff_Ra < 5.0
        println("Good agreement with published results.")
    else
        println("Moderate agreement. Consider increasing resolution.")
    end
else
    println("No successful results obtained.")
    println("Try adjusting Ra_bracket or increasing resolution.")
end

println()
println("="^70)
