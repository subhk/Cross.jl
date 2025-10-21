# =============================================================================
#  Benchmark Tests for Onset of Convection
#  Compares against Dormy et al. (2004) Table 1 values
# =============================================================================

using Cross
using Printf

"""
Benchmark against Table 1 in docs/poloidal_toroidal_derivation.tex
which comes from Dormy et al. (2004) Table 5.
"""
function benchmark_dormy2004()
    println("="^70)
    println("Benchmarking against Dormy et al. (2004)")
    println("Aspect ratio χ = 0.35, Pr = 1")
    println("="^70)
    println()

    # Test cases from Dormy et al. (2004) Table 5
    # Columns: E, Ra_c^D, m_c^D, ω_c^D
    test_cases = [
        (4.734e-5, 1.6525e6, 9,  -0.011003),
        (4.734e-6, 2.6279e7, 19, -0.005691),
        (4.734e-7, 4.6180e8, 40, -0.002804)
    ]

    χ = 0.35
    Pr = 1.0

    results = []

    println(@sprintf("%-12s %-15s %-15s %-8s %-15s %-10s %-10s",
                    "E", "Ra_c (Dormy)", "Ra_c (Cross)", "m_c", "ω_c (Dormy)",
                    "ω_c (Cross)", "%ΔRa"))
    println("-"^100)

    for (E, Ra_c_dormy, m_c_dormy, ω_c_dormy) in test_cases
        m = Int(m_c_dormy)

        # Determine appropriate numerical resolution
        if E >= 1e-5
            lmax = 40
            Nr = 50
        elseif E >= 1e-6
            lmax = 60
            Nr = 64
        else
            lmax = 100
            Nr = 96
        end

        try
            # Find critical Rayleigh number
            Ra_c, ω_c, vec = find_critical_rayleigh(
                E, Pr, χ, m, lmax, Nr;
                Ra_guess=Ra_c_dormy,
                Ra_bracket=(Ra_c_dormy*0.5, Ra_c_dormy*1.5),
                mechanical_bc=:no_slip,
                thermal_bc=:fixed_temperature
            )

            # Calculate percentage difference
            pct_diff_Ra = 100 * abs(Ra_c - Ra_c_dormy) / Ra_c_dormy
            pct_diff_ω = 100 * abs(ω_c - ω_c_dormy) / abs(ω_c_dormy)

            println(@sprintf("%-12.3e %-15.6e %-15.6e %-8d %-15.6f %-10.6f %-10.3f",
                           E, Ra_c_dormy, Ra_c, m, ω_c_dormy, ω_c, pct_diff_Ra))

            push!(results, (E=E, Ra_c=Ra_c, m_c=m, ω_c=ω_c,
                           pct_diff_Ra=pct_diff_Ra, pct_diff_ω=pct_diff_ω))

        catch e
            println(@sprintf("%-12.3e %-15s %-15s %-8d %-15s %-10s %-10s",
                           E, "FAILED", "-", m, "-", "-", "-"))
            println("   Error: ", e)
        end
    end

    println()
    println("="^70)
    println("Benchmark Complete")
    println("="^70)

    return results
end

# """
# Quick test at moderate Ekman number
# """
# function quick_test()
#     println("Quick test at E = 1e-4, χ = 0.35")
#     println()

#     E = 1e-4
#     χ = 0.35
#     Pr = 1.0
#     m = 9
#     lmax = 25
#     Nr = 32

#     # Initial guess
#     Ra_guess = 5e5

#     println("Finding critical Rayleigh number...")
#     Ra_c, ω_c, vec = find_critical_rayleigh(
#         E, Pr, χ, m, lmax, Nr;
#         Ra_guess=Ra_guess,
#         mechanical_bc=:no_slip,
#         thermal_bc=:fixed_temperature
#     )

#     println()
#     println("Results:")
#     println("  Ra_c = ", Ra_c)
#     println("  ω_c  = ", ω_c)
#     println("  m_c  = ", m)
#     println()

#     return Ra_c, ω_c
# end

# """
# Test growth rate calculation
# """
# function test_growth_rate()
#     println("Testing growth rate calculation")
#     println()

#     E = 1e-4
#     χ = 0.35
#     Pr = 1.0
#     m = 9
#     lmax = 25
#     Nr = 32

#     # Test at different Rayleigh numbers
#     Ra_values = [1e5, 5e5, 1e6, 2e6]

#     println(@sprintf("%-12s %-12s %-12s", "Ra", "σ", "ω"))
#     println("-"^40)

#     for Ra in Ra_values
#         params = OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=m, lmax=lmax, Nr=Nr,
#                             mechanical_bc=:no_slip,
#                             thermal_bc=:fixed_temperature)
#         op = LinearStabilityOperator(params)

#         σ, ω, vec = find_growth_rate(op)

#         println(@sprintf("%-12.3e %-12.6f %-12.6f", Ra, σ, ω))
#     end

#     println()
# end

# Run tests if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running onset of convection benchmark tests")
    println()

    # println("Test 1: Quick test")
    # quick_test()

    # println("\nTest 2: Growth rate vs Rayleigh number")
    # test_growth_rate()

    println("\nTest 3: Full benchmark against Dormy et al. (2004)")
    benchmark_dormy2004()
end
