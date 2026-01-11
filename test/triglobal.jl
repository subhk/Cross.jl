# =============================================================================
#  Tests for Tri-Global Stability Analysis
# =============================================================================

using Test
using Cross
using LinearAlgebra

@testset "Triglobal Stability" begin

    # =========================================================================
    # Test 1: Gaunt Coefficient / Spherical Harmonic Coupling
    # =========================================================================
    @testset "Gaunt Coefficients" begin
        # Import the internal function for testing
        compute_sh_coupling = Cross.compute_sh_coupling_coefficient

        # Selection rule: m1 + m2 must equal m3
        @test compute_sh_coupling(1, 0, 1, 1, 2, 0) == 0.0  # 0 + 1 ≠ 0
        @test compute_sh_coupling(1, 1, 1, 1, 2, 0) == 0.0  # 1 + 1 ≠ 0

        # Triangle inequality: |ℓ1 - ℓ2| ≤ ℓ3 ≤ ℓ1 + ℓ2
        @test compute_sh_coupling(1, 0, 1, 0, 5, 0) == 0.0  # 5 > 1 + 1

        # Parity: ℓ1 + ℓ2 + ℓ3 must be even
        @test compute_sh_coupling(1, 0, 1, 0, 1, 0) == 0.0  # 1 + 1 + 1 = 3 (odd)

        # Valid coupling: Y_10 × Y_10 → Y_20
        # ℓ1=1, m1=0, ℓ2=1, m2=0, ℓ3=2, m3=0
        # Selection rules: 0+0=0 ✓, |1-1|=0 ≤ 2 ≤ 1+1=2 ✓, 1+1+2=4 even ✓
        g_110_110_200 = compute_sh_coupling(1, 0, 1, 0, 2, 0)
        @test abs(g_110_110_200) > 0.0  # Should be non-zero

        # Valid coupling: Y_20 × Y_10 → Y_10
        g_20_10_10 = compute_sh_coupling(2, 0, 1, 0, 1, 0)
        @test abs(g_20_10_10) > 0.0

        # Valid coupling: Y_11 × Y_2,-1 → Y_10
        # m1=1, m2=-1, m3=0: 1 + (-1) = 0 ✓
        g_11_2m1_10 = compute_sh_coupling(1, 1, 2, -1, 1, 0)
        @test abs(g_11_2m1_10) > 0.0

        # m constraint: |m| must be ≤ ℓ
        @test compute_sh_coupling(1, 2, 1, 0, 2, 2) == 0.0  # |2| > 1

        # Known analytical value: ∫ Y_00 Y_00 Y_00 dΩ = 1/√(4π)
        # But our function computes ∫ Y*_00 Y_00 Y_00 dΩ with Gaunt convention
        g_000 = compute_sh_coupling(0, 0, 0, 0, 0, 0)
        expected_g000 = 1.0 / sqrt(4π)
        @test isapprox(g_000, expected_g000, rtol=1e-10)
    end

    # =========================================================================
    # Test 1b: Unweighted Spherical Harmonic Coupling (for 1/sinθ terms)
    # =========================================================================
    @testset "Unweighted SH Coupling" begin
        compute_sh_unweighted = Cross.compute_sh_coupling_unweighted
        compute_sh_gaunt = Cross.compute_sh_coupling_coefficient

        # Selection rules should be the same as Gaunt
        @test compute_sh_unweighted(1, 0, 1, 1, 2, 0) == 0.0  # m selection
        @test compute_sh_unweighted(1, 0, 1, 0, 5, 0) == 0.0  # triangle
        @test compute_sh_unweighted(1, 0, 1, 0, 1, 0) == 0.0  # parity

        # Valid couplings should be non-zero
        g_unw_110_110_200 = compute_sh_unweighted(1, 0, 1, 0, 2, 0)
        @test abs(g_unw_110_110_200) > 0.0

        # The unweighted integral should differ from Gaunt for non-trivial cases
        # For Y_10 × Y_10 → Y_20, the two integrals should be different
        g_gaunt = compute_sh_gaunt(1, 0, 1, 0, 2, 0)
        g_unweighted = compute_sh_unweighted(1, 0, 1, 0, 2, 0)

        # They should have the same sign but different magnitudes
        @test sign(g_gaunt) == sign(g_unweighted)
        @test !isapprox(abs(g_gaunt), abs(g_unweighted), rtol=0.01)

        # For Y_00 × Y_00 → Y_00 (all constants), both should give same result
        # since the 1/sinθ factor cancels with the measure
        g_gaunt_000 = compute_sh_gaunt(0, 0, 0, 0, 0, 0)
        g_unweighted_000 = compute_sh_unweighted(0, 0, 0, 0, 0, 0)
        # The unweighted integral: ∫ Y_00³ dθ dφ = (1/(4π))^(3/2) × ∫ dθ dφ
        #                                        = (1/(4π))^(3/2) × 2π × π
        # This differs from Gaunt by a factor related to the measure
        @test abs(g_unweighted_000) > 0.0
        @test isfinite(g_unweighted_000)

        # Test normalized associated Legendre function
        _norm_legendre = Cross._normalized_associated_legendre

        # P̃_0^0(x) = 1/√(4π) for all x
        @test isapprox(_norm_legendre(0, 0, 0.5), 1/sqrt(4π), rtol=1e-10)
        @test isapprox(_norm_legendre(0, 0, -0.3), 1/sqrt(4π), rtol=1e-10)

        # P̃_1^0(x) = √(3/(4π)) x
        @test isapprox(_norm_legendre(1, 0, 0.5), sqrt(3/(4π)) * 0.5, rtol=1e-10)

        # P̃_1^1 at x=0 (θ=π/2): should be -√(3/(8π))
        # At equator sinθ = 1, so P_1^1 = -sinθ = -1
        p11_equator = _norm_legendre(1, 1, 0.0)
        @test abs(p11_equator) > 0.0  # Should be non-zero at equator

        # Orthogonality check: ∫ P̃_ℓ^m P̃_ℓ'^m dΩ should be δ_{ℓℓ'}
        # We can't easily test this without integration, but we can verify
        # that values are reasonable
        @test isfinite(_norm_legendre(5, 3, 0.7))
        @test isfinite(_norm_legendre(10, 5, -0.2))
    end

    # =========================================================================
    # Test 2: Mode Coupling Structure
    # =========================================================================
    @testset "Mode Coupling Structure" begin
        get_coupling_modes = Cross.get_coupling_modes

        # Basic state with m_bs = 2
        m_range = -4:4

        # Mode 0 couples to -2, 0, 2
        coupled_0 = get_coupling_modes(0, 2, m_range)
        @test -2 in coupled_0
        @test 0 in coupled_0
        @test 2 in coupled_0

        # Mode 1 couples to -1, 1, 3
        coupled_1 = get_coupling_modes(1, 2, m_range)
        @test -1 in coupled_1
        @test 1 in coupled_1
        @test 3 in coupled_1

        # Edge case: mode at boundary only couples partially
        coupled_4 = get_coupling_modes(4, 2, m_range)
        @test 2 in coupled_4
        @test 4 in coupled_4
        @test !(6 in coupled_4)  # Outside range
    end

    # =========================================================================
    # Test 3: DOF Counting Consistency
    # =========================================================================
    @testset "DOF Counting" begin
        E = 1e-4
        Pr = 1.0
        Ra = 1e5
        χ = 0.35
        Nr = 16
        lmax = 10

        # Create a simple 3D basic state
        cd = ChebyshevDiffn(Nr, [χ, 1.0], 2)
        amplitudes = Dict((2, 0) => 0.1, (2, 2) => 0.05)
        bs3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 2, amplitudes)

        # Set up triglobal problem
        m_range = -2:2
        params = TriglobalParams(
            E = E, Pr = Pr, Ra = Ra, χ = χ,
            m_range = m_range,
            lmax = lmax,
            Nr = Nr,
            basic_state_3d = bs3d,
            mechanical_bc = :no_slip,
            thermal_bc = :fixed_temperature
        )

        problem = setup_coupled_mode_problem(params)

        # Verify block indices are contiguous and cover all DOFs
        all_indices = Int[]
        for m in m_range
            append!(all_indices, collect(problem.block_indices[m]))
        end
        sort!(all_indices)

        @test all_indices == collect(1:problem.total_dofs)

        # Verify each block has expected size
        for m in m_range
            num_ell = lmax - abs(m) + 1
            expected_block_size = num_ell * (3 * Nr - 8)
            actual_block_size = length(problem.block_indices[m])
            @test actual_block_size == expected_block_size
        end
    end

    # =========================================================================
    # Test 4: Axisymmetric Limit (No Mode Coupling)
    # =========================================================================
    @testset "Axisymmetric Limit" begin
        E = 1e-4
        Pr = 1.0
        Ra = 1e5
        χ = 0.35
        Nr = 16
        lmax = 10

        # Create axisymmetric basic state (m=0 only)
        cd = ChebyshevDiffn(Nr, [χ, 1.0], 2)
        amplitudes = Dict((2, 0) => 0.1)  # Only axisymmetric mode
        bs3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 0, amplitudes)

        # Set up problem
        m_range = -2:2
        params = TriglobalParams(
            E = E, Pr = Pr, Ra = Ra, χ = χ,
            m_range = m_range,
            lmax = lmax,
            Nr = Nr,
            basic_state_3d = bs3d,
            mechanical_bc = :no_slip,
            thermal_bc = :fixed_temperature
        )

        problem = setup_coupled_mode_problem(params)

        # With axisymmetric basic state, there should be no mode coupling
        # (all_m_bs should be empty since m_bs = 0 is not counted as "non-axisymmetric")
        @test isempty(problem.all_m_bs)

        # Each mode should only couple to itself
        for m in m_range
            @test problem.coupling_graph[m] == [m]
        end
    end

    # =========================================================================
    # Test 5: Eigenvalue Solver Smoke Test
    # =========================================================================
    @testset "Eigenvalue Solver" begin
        # Use small problem for fast testing
        E = 1e-3
        Pr = 1.0
        Ra = 1e4
        χ = 0.35
        Nr = 12
        lmax = 6

        # Create basic state
        cd = ChebyshevDiffn(Nr, [χ, 1.0], 2)
        amplitudes = Dict((2, 0) => 0.1, (2, 2) => 0.02)
        bs3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 2, amplitudes)

        # Set up small problem
        m_range = -1:1  # Only 3 modes for speed
        params = TriglobalParams(
            E = E, Pr = Pr, Ra = Ra, χ = χ,
            m_range = m_range,
            lmax = lmax,
            Nr = Nr,
            basic_state_3d = bs3d,
            mechanical_bc = :no_slip,
            thermal_bc = :fixed_temperature
        )

        # Solve eigenvalue problem (smoke test)
        eigenvalues, eigenvectors = solve_triglobal_eigenvalue_problem(
            params;
            σ_target = 0.0,
            nev = 4,
            verbose = false
        )

        # Basic sanity checks
        @test length(eigenvalues) >= 1
        @test size(eigenvectors, 1) > 0

        # Eigenvalues should be sorted by real part (descending)
        for i in 1:(length(eigenvalues)-1)
            @test real(eigenvalues[i]) >= real(eigenvalues[i+1]) - 1e-10
        end

        # Eigenvalues should be finite
        @test all(isfinite.(eigenvalues))
    end

    # =========================================================================
    # Test 6: Basic State Mode Scale Factor
    # =========================================================================
    @testset "Basic State Mode Scale" begin
        _basic_state_mode_scale = Cross._basic_state_mode_scale

        # m = 0: scale should be 1
        @test _basic_state_mode_scale(0, Float64) == 1.0

        # m ≠ 0: scale should be 1/√2
        @test isapprox(_basic_state_mode_scale(2, Float64), 1/sqrt(2), rtol=1e-10)
        @test isapprox(_basic_state_mode_scale(-2, Float64), 1/sqrt(2), rtol=1e-10)

        # Odd negative m: should have negative phase
        @test _basic_state_mode_scale(-1, Float64) < 0
        @test _basic_state_mode_scale(-3, Float64) < 0

        # Even negative m: should have positive phase
        @test _basic_state_mode_scale(-2, Float64) > 0
        @test _basic_state_mode_scale(-4, Float64) > 0
    end

    # =========================================================================
    # Test 7: Theta Derivative Coefficients
    # =========================================================================
    @testset "Theta Derivative Coefficients" begin
        _theta_derivative_coeff = Cross._theta_derivative_coeff

        # ℓ = 0: derivative should be zero (constant)
        c_plus, c_minus = _theta_derivative_coeff(0, 0)
        @test c_plus == 0.0
        @test c_minus == 0.0

        # ℓ = 1, m = 0: should have non-zero c_plus (couples to ℓ=2)
        c_plus, c_minus = _theta_derivative_coeff(1, 0)
        @test c_plus != 0.0
        @test c_minus == 0.0  # No ℓ-1 = 0 coupling for m=0

        # ℓ = 2, m = 0: should have both non-zero
        c_plus, c_minus = _theta_derivative_coeff(2, 0)
        @test c_plus != 0.0
        @test c_minus != 0.0

        # Invalid case: ℓ < |m|
        c_plus, c_minus = _theta_derivative_coeff(1, 2)
        @test c_plus == 0.0
        @test c_minus == 0.0
    end

end  # @testset "Triglobal Stability"
