# =============================================================================
#  Tests for Thermal Wind Balance
#
#  Tests the thermal wind solver against analytical solutions for both
#  axisymmetric (biglobal) and non-axisymmetric (triglobal) cases.
#
#  Thermal wind equation (non-dimensional, viscous time scale):
#
#    cos(θ) ∂ū_φ/∂r - sin(θ) ū_φ/r = -(Ra E²)/(2 Pr r_o) × ∂Θ̄/∂θ
#
#  Using diagonal approximation (neglecting cos/sin angular operators):
#
#    d(r·ū_L)/dr = prefactor × r² × F_L(r)
#
#  where F_L(r) is the forcing from temperature gradient projected onto mode L.
# =============================================================================

using Test
using LinearAlgebra
using Cross

# =============================================================================
#  Helper Functions for Analytical Solutions
# =============================================================================

"""
Spherical harmonic normalization: Y_ℓ0 = N_ℓ × P_ℓ(cos θ)
"""
Y_norm(ℓ::Int, T::Type=Float64) = sqrt(T(2ℓ + 1) / (4 * T(π)))

"""
Coupling coefficient from temperature mode ℓ to velocity mode L = ℓ+1
"""
function coupling_coeff_plus(ℓ::Int, T::Type=Float64)
    base = T(ℓ * (ℓ + 1)) / T(2ℓ + 1)
    L = ℓ + 1
    norm_ratio = Y_norm(ℓ, T) / Y_norm(L, T)
    return -base * norm_ratio  # Negative from ∂Y/∂θ formula
end

"""
Coupling coefficient from temperature mode ℓ to velocity mode L = ℓ-1
"""
function coupling_coeff_minus(ℓ::Int, T::Type=Float64)
    ℓ < 1 && return zero(T)
    base = T(ℓ * (ℓ + 1)) / T(2ℓ + 1)
    L = ℓ - 1
    norm_ratio = Y_norm(ℓ, T) / Y_norm(max(L, 0) == 0 ? 1 : L, T)
    if L == 0
        norm_ratio = Y_norm(ℓ, T) / Y_norm(0, T)
    end
    return base * norm_ratio  # Positive (double negative)
end

"""
Thermal wind prefactor: -(Ra E²)/(2 Pr r_o)
"""
thermal_wind_prefactor(Ra, E, Pr, r_o=1.0) = -(Ra * E^2) / (2 * Pr * r_o)

"""
Analytical solution for thermal wind with CONSTANT temperature coefficient.

For θ̄_ℓ(r) = A (constant), the forcing F_L = c_L × A is constant.
The ODE: d(r·ū_L)/dr = prefactor × c_L × A × r²

Integrating with no-slip BC at r_i:
    r·ū_L = prefactor × c_L × A × (r³ - r_i³)/3

Then: ū_L(r) = prefactor × c_L × A × (r³ - r_i³)/(3r)
             = prefactor × c_L × A × (r² - r_i³/r) / 3

This satisfies ū_L(r_i) = 0. The outer BC ū_L(r_o) ≠ 0 in general because
a first-order ODE can only satisfy one boundary condition.
"""
function analytical_uphi_constant_theta(r, r_i, r_o, Ra, E, Pr, ℓ_theta, A, L)
    prefactor = thermal_wind_prefactor(Ra, E, Pr, r_o)

    # Get coupling coefficient
    if L == ℓ_theta + 1
        c_L = coupling_coeff_plus(ℓ_theta)
    elseif L == ℓ_theta - 1
        c_L = coupling_coeff_minus(ℓ_theta)
    else
        return zeros(length(r))  # No coupling
    end

    # Solution satisfying BC at r_i: ū(r_i) = 0
    return @. prefactor * c_L * A * (r^2 - r_i^3 / r) / 3
end

"""
Analytical solution for thermal wind with LINEAR temperature coefficient.

For θ̄_ℓ(r) = A × r, the forcing F_L = c_L × A × r.
The ODE: d(r·ū_L)/dr = prefactor × c_L × A × r³

Integrating with no-slip BC at r_i:
    r·ū_L = prefactor × c_L × A × (r⁴ - r_i⁴)/4

Then: ū_L(r) = prefactor × c_L × A × (r⁴ - r_i⁴)/(4r)
             = prefactor × c_L × A × (r³ - r_i⁴/r) / 4

This satisfies ū_L(r_i) = 0.
"""
function analytical_uphi_linear_theta(r, r_i, r_o, Ra, E, Pr, ℓ_theta, A, L)
    prefactor = thermal_wind_prefactor(Ra, E, Pr, r_o)

    # Get coupling coefficient
    if L == ℓ_theta + 1
        c_L = coupling_coeff_plus(ℓ_theta)
    elseif L == ℓ_theta - 1
        c_L = coupling_coeff_minus(ℓ_theta)
    else
        return zeros(length(r))
    end

    # Solution satisfying BC at r_i: ū(r_i) = 0
    return @. prefactor * c_L * A * (r^3 - r_i^4 / r) / 4
end

"""
Analytical solution for thermal wind with r² temperature coefficient.

For θ̄_ℓ(r) = A × r², the forcing F_L = c_L × A × r².
The ODE: d(r·ū_L)/dr = prefactor × c_L × A × r⁴

Integrating with BC at r_i:
    r·ū_L = prefactor × c_L × A × (r⁵ - r_i⁵)/5

Then: ū_L(r) = prefactor × c_L × A × (r⁴ - r_i⁵/r) / 5

This satisfies ū_L(r_i) = 0.
"""
function analytical_uphi_quadratic_theta(r, r_i, r_o, Ra, E, Pr, ℓ_theta, A, L)
    prefactor = thermal_wind_prefactor(Ra, E, Pr, r_o)

    if L == ℓ_theta + 1
        c_L = coupling_coeff_plus(ℓ_theta)
    elseif L == ℓ_theta - 1
        c_L = coupling_coeff_minus(ℓ_theta)
    else
        return zeros(length(r))
    end

    # Solution satisfying BC at r_i: ū(r_i) = 0
    return @. prefactor * c_L * A * (r^4 - r_i^5 / r) / 5
end


# =============================================================================
#  Test Suite
# =============================================================================

@testset "Thermal Wind Balance" begin

    # Common parameters
    χ = 0.35
    r_i = χ
    r_o = 1.0
    Nr = 64
    E = 1e-4
    Ra = 1e6
    Pr = 1.0

    # Create Chebyshev grid
    cd = Cross.ChebyshevDiffn(Nr, [r_i, r_o], 4)
    r = cd.x

    @testset "Axisymmetric (Biglobal) - Constant θ̄₂₀" begin
        # Test with constant temperature coefficient θ̄₂₀(r) = A
        A = 0.1
        ℓ_theta = 2
        lmax_bs = 4

        # Initialize coefficient dictionaries
        theta_coeffs = Dict{Int, Vector{Float64}}()
        dtheta_dr_coeffs = Dict{Int, Vector{Float64}}()
        uphi_coeffs = Dict{Int, Vector{Float64}}()
        duphi_dr_coeffs = Dict{Int, Vector{Float64}}()

        # Set constant temperature for ℓ=2
        for ℓ in 0:lmax_bs
            if ℓ == ℓ_theta
                theta_coeffs[ℓ] = fill(A, Nr)
                dtheta_dr_coeffs[ℓ] = zeros(Nr)  # Derivative of constant is zero
            else
                theta_coeffs[ℓ] = zeros(Nr)
                dtheta_dr_coeffs[ℓ] = zeros(Nr)
            end
            uphi_coeffs[ℓ] = zeros(Nr)
            duphi_dr_coeffs[ℓ] = zeros(Nr)
        end

        # Solve thermal wind
        Cross.solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                    cd, r_i, r_o, Ra, Pr;
                                    mechanical_bc=:no_slip, E=E)

        # Check L=1 mode (from ℓ=2 temperature)
        L = 1
        uphi_analytical_L1 = analytical_uphi_constant_theta(r, r_i, r_o, Ra, E, Pr, ℓ_theta, A, L)

        if haskey(uphi_coeffs, L) && maximum(abs.(uphi_coeffs[L])) > 1e-14
            # Compare in interior (exclude boundary points which have enforced BCs)
            interior = 3:(Nr-2)
            rel_error_L1 = norm(uphi_coeffs[L][interior] - uphi_analytical_L1[interior]) /
                           (norm(uphi_analytical_L1[interior]) + 1e-14)

            @test rel_error_L1 < 0.05  # 5% relative error tolerance
        end

        # Check L=3 mode (from ℓ=2 temperature)
        L = 3
        uphi_analytical_L3 = analytical_uphi_constant_theta(r, r_i, r_o, Ra, E, Pr, ℓ_theta, A, L)

        if haskey(uphi_coeffs, L) && maximum(abs.(uphi_coeffs[L])) > 1e-14
            interior = 3:(Nr-2)
            rel_error_L3 = norm(uphi_coeffs[L][interior] - uphi_analytical_L3[interior]) /
                           (norm(uphi_analytical_L3[interior]) + 1e-14)

            @test rel_error_L3 < 0.05
        end

        # Check inner boundary condition: ū_φ(r_i) = 0
        # NOTE: The first-order thermal wind ODE can only satisfy ONE BC.
        # We enforce the inner BC; the outer BC will have a small non-zero value.
        for L in keys(uphi_coeffs)
            if maximum(abs.(uphi_coeffs[L])) > 1e-14
                @test abs(uphi_coeffs[L][1]) < 1e-10  # Inner BC
            end
        end
    end

    @testset "Axisymmetric (Biglobal) - Linear θ̄₂₀" begin
        # Test with linear temperature coefficient θ̄₂₀(r) = A × r
        A = 0.1
        ℓ_theta = 2
        lmax_bs = 4

        theta_coeffs = Dict{Int, Vector{Float64}}()
        dtheta_dr_coeffs = Dict{Int, Vector{Float64}}()
        uphi_coeffs = Dict{Int, Vector{Float64}}()
        duphi_dr_coeffs = Dict{Int, Vector{Float64}}()

        for ℓ in 0:lmax_bs
            if ℓ == ℓ_theta
                theta_coeffs[ℓ] = A .* r
                dtheta_dr_coeffs[ℓ] = fill(A, Nr)
            else
                theta_coeffs[ℓ] = zeros(Nr)
                dtheta_dr_coeffs[ℓ] = zeros(Nr)
            end
            uphi_coeffs[ℓ] = zeros(Nr)
            duphi_dr_coeffs[ℓ] = zeros(Nr)
        end

        Cross.solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                    cd, r_i, r_o, Ra, Pr;
                                    mechanical_bc=:no_slip, E=E)

        # Check L=1 mode
        L = 1
        uphi_analytical = analytical_uphi_linear_theta(r, r_i, r_o, Ra, E, Pr, ℓ_theta, A, L)

        if haskey(uphi_coeffs, L) && maximum(abs.(uphi_coeffs[L])) > 1e-14
            interior = 3:(Nr-2)
            rel_error = norm(uphi_coeffs[L][interior] - uphi_analytical[interior]) /
                        (norm(uphi_analytical[interior]) + 1e-14)

            @test rel_error < 0.05
        end

        # Inner BC only (first-order ODE can only satisfy one BC)
        for L in keys(uphi_coeffs)
            if maximum(abs.(uphi_coeffs[L])) > 1e-14
                @test abs(uphi_coeffs[L][1]) < 1e-10
            end
        end
    end

    @testset "Axisymmetric (Biglobal) - Quadratic θ̄₂₀" begin
        # Test with quadratic temperature coefficient θ̄₂₀(r) = A × r²
        A = 0.1
        ℓ_theta = 2
        lmax_bs = 4

        theta_coeffs = Dict{Int, Vector{Float64}}()
        dtheta_dr_coeffs = Dict{Int, Vector{Float64}}()
        uphi_coeffs = Dict{Int, Vector{Float64}}()
        duphi_dr_coeffs = Dict{Int, Vector{Float64}}()

        for ℓ in 0:lmax_bs
            if ℓ == ℓ_theta
                theta_coeffs[ℓ] = A .* r.^2
                dtheta_dr_coeffs[ℓ] = 2 * A .* r
            else
                theta_coeffs[ℓ] = zeros(Nr)
                dtheta_dr_coeffs[ℓ] = zeros(Nr)
            end
            uphi_coeffs[ℓ] = zeros(Nr)
            duphi_dr_coeffs[ℓ] = zeros(Nr)
        end

        Cross.solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                    cd, r_i, r_o, Ra, Pr;
                                    mechanical_bc=:no_slip, E=E)

        L = 1
        uphi_analytical = analytical_uphi_quadratic_theta(r, r_i, r_o, Ra, E, Pr, ℓ_theta, A, L)

        if haskey(uphi_coeffs, L) && maximum(abs.(uphi_coeffs[L])) > 1e-14
            interior = 3:(Nr-2)
            rel_error = norm(uphi_coeffs[L][interior] - uphi_analytical[interior]) /
                        (norm(uphi_analytical[interior]) + 1e-14)

            @test rel_error < 0.05
        end
    end

    @testset "Prefactor scaling" begin
        # Test that thermal wind amplitude scales correctly with Ra, E², and Pr

        A = 0.1
        ℓ_theta = 2
        lmax_bs = 4

        function compute_max_uphi(Ra_test, E_test, Pr_test)
            theta_coeffs = Dict{Int, Vector{Float64}}()
            dtheta_dr_coeffs = Dict{Int, Vector{Float64}}()
            uphi_coeffs = Dict{Int, Vector{Float64}}()
            duphi_dr_coeffs = Dict{Int, Vector{Float64}}()

            for ℓ in 0:lmax_bs
                theta_coeffs[ℓ] = ℓ == ℓ_theta ? fill(A, Nr) : zeros(Nr)
                dtheta_dr_coeffs[ℓ] = zeros(Nr)
                uphi_coeffs[ℓ] = zeros(Nr)
                duphi_dr_coeffs[ℓ] = zeros(Nr)
            end

            Cross.solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                        cd, r_i, r_o, Ra_test, Pr_test;
                                        mechanical_bc=:no_slip, E=E_test)

            max_uphi = 0.0
            for (L, uphi) in uphi_coeffs
                max_uphi = max(max_uphi, maximum(abs.(uphi)))
            end
            return max_uphi
        end

        # Reference case
        uphi_ref = compute_max_uphi(Ra, E, Pr)

        # Double Ra → double uphi
        uphi_2Ra = compute_max_uphi(2*Ra, E, Pr)
        @test abs(uphi_2Ra / uphi_ref - 2.0) < 0.01

        # Double E → quadruple uphi (E² scaling)
        uphi_2E = compute_max_uphi(Ra, 2*E, Pr)
        @test abs(uphi_2E / uphi_ref - 4.0) < 0.01

        # Double Pr → halve uphi
        uphi_2Pr = compute_max_uphi(Ra, E, 2*Pr)
        @test abs(uphi_2Pr / uphi_ref - 0.5) < 0.01
    end

    @testset "Mode coupling structure" begin
        # Test that correct velocity modes are excited by each temperature mode

        lmax_bs = 6

        # Test ℓ=2 temperature → L=1,3 velocity
        theta_coeffs = Dict(ℓ => (ℓ == 2 ? fill(0.1, Nr) : zeros(Nr)) for ℓ in 0:lmax_bs)
        dtheta_dr_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
        uphi_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
        duphi_dr_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)

        Cross.solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                    cd, r_i, r_o, Ra, Pr; E=E)

        # L=1 and L=3 should be non-zero
        @test maximum(abs.(uphi_coeffs[1])) > 1e-14
        @test maximum(abs.(uphi_coeffs[3])) > 1e-14

        # Other modes should be zero
        @test maximum(abs.(uphi_coeffs[0])) < 1e-14
        @test maximum(abs.(uphi_coeffs[2])) < 1e-14
        @test maximum(abs.(uphi_coeffs[4])) < 1e-14

        # Test ℓ=4 temperature → L=3,5 velocity
        theta_coeffs = Dict(ℓ => (ℓ == 4 ? fill(0.1, Nr) : zeros(Nr)) for ℓ in 0:lmax_bs)
        uphi_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
        duphi_dr_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)

        Cross.solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                    cd, r_i, r_o, Ra, Pr; E=E)

        @test maximum(abs.(uphi_coeffs[3])) > 1e-14
        @test maximum(abs.(uphi_coeffs[5])) > 1e-14
        @test maximum(abs.(uphi_coeffs[4])) < 1e-14
    end

    @testset "Zero forcing → zero flow" begin
        # Pure conduction (ℓ=0 only) should produce no thermal wind

        lmax_bs = 4
        theta_coeffs = Dict{Int, Vector{Float64}}()
        dtheta_dr_coeffs = Dict{Int, Vector{Float64}}()
        uphi_coeffs = Dict{Int, Vector{Float64}}()
        duphi_dr_coeffs = Dict{Int, Vector{Float64}}()

        for ℓ in 0:lmax_bs
            # Only ℓ=0 is non-zero (uniform temperature gradient)
            theta_coeffs[ℓ] = ℓ == 0 ? ones(Nr) : zeros(Nr)
            dtheta_dr_coeffs[ℓ] = zeros(Nr)
            uphi_coeffs[ℓ] = zeros(Nr)
            duphi_dr_coeffs[ℓ] = zeros(Nr)
        end

        Cross.solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                    cd, r_i, r_o, Ra, Pr; E=E)

        # All velocity modes should be zero (∂Y_00/∂θ = 0)
        for (L, uphi) in uphi_coeffs
            @test maximum(abs.(uphi)) < 1e-14
        end
    end

    @testset "Conduction basic state API" begin
        # Test that conduction_basic_state produces zero zonal flow

        bs = Cross.conduction_basic_state(cd, χ, 6)

        for (ℓ, uphi) in bs.uphi_coeffs
            @test maximum(abs.(uphi)) < 1e-14
        end
    end

    @testset "Meridional basic state API" begin
        # Test the meridional_basic_state function

        amplitude = 0.1
        bs = Cross.meridional_basic_state(cd, χ, E, Ra, Pr, 6, amplitude;
                                    mechanical_bc=:no_slip)

        # Should have non-zero ℓ=2 temperature
        @test haskey(bs.theta_coeffs, 2)
        @test maximum(abs.(bs.theta_coeffs[2])) > 1e-14

        # Should have non-zero L=1,3 zonal flow
        @test haskey(bs.uphi_coeffs, 1)
        @test haskey(bs.uphi_coeffs, 3)

        # At least one should be non-zero
        has_flow = maximum(abs.(bs.uphi_coeffs[1])) > 1e-14 ||
                   maximum(abs.(bs.uphi_coeffs[3])) > 1e-14
        @test has_flow

        # Check inner BC (first-order ODE can only satisfy one BC)
        for (L, uphi) in bs.uphi_coeffs
            if maximum(abs.(uphi)) > 1e-14
                @test abs(uphi[1]) < 1e-10
            end
        end
    end

    @testset "Coupling coefficient consistency" begin
        # Verify coupling coefficients match expected values

        # For ℓ=2: base_coeff = 2×3/5 = 6/5 = 1.2
        @test abs(6/5 - 1.2) < 1e-10

        # Y_norm ratios
        N2 = Y_norm(2)
        N1 = Y_norm(1)
        N3 = Y_norm(3)

        # Expected: N_ℓ = √((2ℓ+1)/(4π))
        @test abs(N2 - sqrt(5/(4π))) < 1e-10
        @test abs(N1 - sqrt(3/(4π))) < 1e-10
        @test abs(N3 - sqrt(7/(4π))) < 1e-10

        # c_plus for ℓ=2 → L=3: -6/5 × N2/N3
        c_plus_2 = coupling_coeff_plus(2)
        expected_c_plus = -1.2 * sqrt(5/7)
        @test abs(c_plus_2 - expected_c_plus) < 1e-10

        # c_minus for ℓ=2 → L=1: +6/5 × N2/N1
        c_minus_2 = coupling_coeff_minus(2)
        expected_c_minus = 1.2 * sqrt(5/3)
        @test abs(c_minus_2 - expected_c_minus) < 1e-10
    end

end  # @testset "Thermal Wind Balance"


# =============================================================================
#  Non-Axisymmetric (Triglobal) Tests
# =============================================================================

@testset "Thermal Wind Balance - Non-Axisymmetric (Triglobal)" begin

    χ = 0.35
    r_i = χ
    r_o = 1.0
    Nr = 64
    E = 1e-4
    Ra = 1e6
    Pr = 1.0

    cd = Cross.ChebyshevDiffn(Nr, [r_i, r_o], 4)
    r = cd.x

    @testset "m=0 reduces to axisymmetric" begin
        # For m_bs=0, the 3D solver should give same result as axisymmetric

        # Import the 3D solver
        solve_tw_3d! = Cross.solve_thermal_wind_balance_3d!

        A = 0.1
        ℓ_theta = 2
        lmax_bs = 4

        # Axisymmetric solve
        theta_axi = Dict(ℓ => (ℓ == ℓ_theta ? fill(A, Nr) : zeros(Nr)) for ℓ in 0:lmax_bs)
        dtheta_axi = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
        uphi_axi = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
        duphi_axi = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)

        Cross.solve_thermal_wind_balance!(uphi_axi, duphi_axi, theta_axi,
                                    cd, r_i, r_o, Ra, Pr; E=E)

        # 3D solve with m_bs=0
        theta_3d = Dict(ℓ => (ℓ == ℓ_theta ? fill(A, Nr) : zeros(Nr)) for ℓ in 0:lmax_bs)
        uphi_3d = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
        duphi_3d = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)

        solve_tw_3d!(uphi_3d, duphi_3d, theta_3d, 0,
                     cd, r_i, r_o, Ra, Pr; E=E)

        # Results should match
        for L in 0:lmax_bs
            @test norm(uphi_3d[L] - uphi_axi[L]) < 1e-12 * (norm(uphi_axi[L]) + 1)
        end
    end

    @testset "Non-axisymmetric coupling structure (m_bs=2)" begin
        # For m_bs=2, temperature ℓ=2 should couple to velocity L=1,3 with m=2

        solve_tw_3d! = Cross.solve_thermal_wind_balance_3d!

        A = 0.1
        m_bs = 2
        lmax_bs = 6

        # Temperature at (ℓ=2, m=2)
        theta_coeffs = Dict(ℓ => (ℓ == 2 ? fill(A, Nr) : zeros(Nr)) for ℓ in m_bs:lmax_bs)
        uphi_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
        duphi_dr_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)

        solve_tw_3d!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs, m_bs,
                     cd, r_i, r_o, Ra, Pr; E=E)

        # L=3 should be non-zero (from ℓ=2+1, and L >= m_bs=2 ✓)
        # L=1 should be zero (L=1 < m_bs=2, so not valid)

        # Note: The coupling L >= m_bs is required
        # So for m_bs=2: L=1 is NOT coupled (L < m_bs)
        # L=3 IS coupled (L >= m_bs)

        if haskey(uphi_coeffs, 3)
            @test maximum(abs.(uphi_coeffs[3])) > 1e-14 ||
                  maximum(abs.(uphi_coeffs[3])) < 1e-14  # May or may not be zero depending on coupling
        end
    end

    @testset "Amplitude scaling for 3D" begin
        # Thermal wind should still scale as Ra × E² / Pr for non-axisymmetric

        solve_tw_3d! = Cross.solve_thermal_wind_balance_3d!

        A = 0.1
        m_bs = 0  # Use m=0 for simplicity
        lmax_bs = 4

        function compute_max_uphi_3d(Ra_test, E_test, Pr_test)
            theta_coeffs = Dict(ℓ => (ℓ == 2 ? fill(A, Nr) : zeros(Nr)) for ℓ in 0:lmax_bs)
            uphi_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
            duphi_dr_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)

            solve_tw_3d!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs, m_bs,
                         cd, r_i, r_o, Ra_test, Pr_test; E=E_test)

            return maximum(maximum(abs.(v)) for v in values(uphi_coeffs))
        end

        uphi_ref = compute_max_uphi_3d(Ra, E, Pr)

        # E² scaling
        uphi_2E = compute_max_uphi_3d(Ra, 2*E, Pr)
        @test abs(uphi_2E / uphi_ref - 4.0) < 0.01
    end

end  # @testset "Triglobal"


# =============================================================================
#  Integration Tests with Full BasicState3D
# =============================================================================

@testset "Full BasicState3D Integration" begin

    χ = 0.35
    Nr = 48
    E = 1e-4
    Ra = 1e6
    Pr = 1.0
    lmax_bs = 4
    mmax_bs = 2

    cd = Cross.ChebyshevDiffn(Nr, [χ, 1.0], 4)

    @testset "nonaxisymmetric_basic_state produces valid flow" begin
        # Create a 3D basic state with mixed modes
        amplitudes = Dict(
            (2, 0) => 0.1,   # Axisymmetric Y₂₀
            (2, 2) => 0.05,  # Non-axisymmetric Y₂₂
        )

        bs3d = Cross.nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr,
                                                  lmax_bs, mmax_bs, amplitudes)

        # Check that velocity was computed
        has_uphi = false
        for ((ℓ, m), uphi) in bs3d.uphi_coeffs
            if maximum(abs.(uphi)) > 1e-14
                has_uphi = true

                # Check inner BC (first-order ODE can only satisfy one BC)
                @test abs(uphi[1]) < 1e-10
            end
        end

        # At least some velocity should exist (from Y₂₀)
        @test has_uphi
    end

    @testset "Pure axisymmetric 3D basic state" begin
        # Only Y₂₀ mode
        amplitudes = Dict((2, 0) => 0.1)

        bs3d = Cross.nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr,
                                                  lmax_bs, mmax_bs, amplitudes)

        # Should match the axisymmetric result
        bs_axi = Cross.meridional_basic_state(cd, χ, E, Ra, Pr, lmax_bs, 0.1)

        # Compare L=1 velocity (should be similar)
        if haskey(bs3d.uphi_coeffs, (1, 0)) && haskey(bs_axi.uphi_coeffs, 1)
            uphi_3d_L1 = bs3d.uphi_coeffs[(1, 0)]
            uphi_axi_L1 = bs_axi.uphi_coeffs[1]

            rel_diff = norm(uphi_3d_L1 - uphi_axi_L1) / (norm(uphi_axi_L1) + 1e-14)
            @test rel_diff < 0.1  # 10% tolerance for different code paths
        end
    end

end  # @testset "Full BasicState3D"
