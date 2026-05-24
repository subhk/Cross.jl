# Regression tests for the public onset / biglobal entry points.
#
# These functions previously failed at call time: their keyword-only `where {T}`
# signatures could not infer T (UndefVarError: T not defined), and
# find_global_critical_onset additionally used findmin over a KeySet. The tests
# below are lightweight smoke tests — they only assert the functions run and
# return finite results — so the API can never silently regress to broken again.

using Test
using Cross

@testset "Public onset/biglobal API runs (keyword-T regression)" begin
    E = 4.225e-4; Pr = 1.0; χ = 0.35   # = literature Ek_d=1e-3 (r_o-based)

    @testset "find_critical_Ra_onset" begin
        Ra_c, ω_c, _ = Cross.find_critical_Ra_onset(
            E=E, Pr=Pr, χ=χ, m=4, lmax=16, Nr=24,
            Ra_guess=5.6e4, Ra_bracket=(1e4, 2e5), nev=6)
        @test isfinite(Ra_c) && Ra_c > 0
        @test isfinite(ω_c)
    end

    @testset "find_global_critical_onset" begin
        m_c, Ra_c, ω_c, _ = Cross.find_global_critical_onset(
            E=E, Pr=Pr, χ=χ, lmax=16, Nr=24, m_range=3:5,
            Ra_guess=5.6e4, verbose=false)
        @test m_c in 3:5
        @test isfinite(Ra_c) && Ra_c > 0
        @test isfinite(ω_c)
    end

    @testset "compare_onset_vs_biglobal" begin
        r = Cross.compare_onset_vs_biglobal(
            E=E, Pr=Pr, χ=χ, m=4, lmax=16, Nr=24, Ra=5.6e4,
            basic_state_amplitude=0.1, verbose=false)
        @test isfinite(r.Δσ) && isfinite(r.Δω)
    end

    @testset "sweep_thermal_wind_amplitude" begin
        rs = Cross.sweep_thermal_wind_amplitude(
            E=E, Pr=Pr, χ=χ, m=4, lmax=16, Nr=24, Ra=5.6e4,
            amplitudes=[0.0, 0.1], verbose=false)
        @test length(rs) == 2
        @test all(isfinite(r.σ) for r in rs)
    end

    @testset "mixed-precision scalar inputs promote cleanly" begin
        # Int Pr, Int-ish args must not break type inference
        Ra_c, ω_c, _ = Cross.find_critical_Ra_onset(
            E=1e-3, Pr=1, χ=0.35, m=2, lmax=12, Nr=16,
            Ra_guess=3e4, Ra_bracket=(5e3, 1e5), nev=4)
        @test isfinite(Ra_c) && Ra_c > 0
    end
end
