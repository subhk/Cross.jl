using Test
using Cross
using Logging

function _capture_stdout(f)
    mktemp() do _, io
        redirect_stdout(io) do
            f()
        end
        flush(io)
        seekstart(io)
        read(io, String)
    end
end

function _basic_state_3d_fixture(params; r=ChebyshevDiffn(params.Nr, [params.χ, 1.0], 1).x,
                                 coefficient_length=params.Nr)
    coeffs = Dict{Tuple{Int,Int},Vector{Float64}}(
        (0, 0) => zeros(Float64, coefficient_length),
    )
    return BasicState3D{Float64}(
        lmax_bs=2, mmax_bs=0, Nr=params.Nr,
        r=collect(Float64, r),
        theta_coeffs=copy(coeffs),
        dtheta_dr_coeffs=copy(coeffs),
        ur_coeffs=copy(coeffs),
        utheta_coeffs=copy(coeffs),
        uphi_coeffs=copy(coeffs),
        dur_dr_coeffs=copy(coeffs),
        dutheta_dr_coeffs=copy(coeffs),
        duphi_dr_coeffs=copy(coeffs)
    )
end

@testset "BiglobalProblem construction and validation" begin
    # Create a valid basic state using the conduction profile
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    cd = ChebyshevDiffn(params.Nr, [params.χ, 1.0], 4)
    bs = conduction_basic_state(cd, params.χ, 6)

    problem = BiglobalProblem(params, bs)
    @test problem.params === params
    @test problem.basic_state === bs

    # Show works
    output = sprint(show, MIME("text/plain"), problem)
    @test occursin("BiglobalProblem", output)

    # Validation: Nr mismatch between params and basic state
    params_bad = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=32)
    @test_throws ArgumentError BiglobalProblem(params_bad, bs)

    # Validation: same Nr but incompatible radial grid
    cd_wrong_grid = ChebyshevDiffn(params.Nr, [0.40, 1.0], 4)
    bs_wrong_grid = conduction_basic_state(cd_wrong_grid, 0.40, 6)
    @test_throws ArgumentError BiglobalProblem(params, bs_wrong_grid)
end

@testset "TriglobalProblem construction and validation" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=0, lmax=10, Nr=16)

    bs3d = _basic_state_3d_fixture(params)

    problem = TriglobalProblem(params, bs3d, -2:2)
    @test problem.params === params
    @test problem.m_range == -2:2

    # Show works
    output = sprint(show, MIME("text/plain"), problem)
    @test occursin("TriglobalProblem", output)

    # Validation: empty m_range
    @test_throws ArgumentError TriglobalProblem(params, bs3d, 1:0)

    # Validation: same Nr but incompatible radial grid
    wrong_grid = ChebyshevDiffn(params.Nr, [0.40, 1.0], 1).x
    @test_throws ArgumentError TriglobalProblem(params, _basic_state_3d_fixture(params; r=wrong_grid), 0:2)

    # Validation: coefficient vectors must match the radial resolution
    @test_throws ArgumentError TriglobalProblem(params, _basic_state_3d_fixture(params; coefficient_length=params.Nr - 1), 0:2)
end

@testset "MHDProblem construction" begin
    params = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, ricb=0.35,
                       m=1, lmax=6, symm=1, N=8)
    problem = MHDProblem(params)
    @test problem.basic_state === nothing

    output = sprint(show, MIME("text/plain"), problem)
    @test occursin("MHDProblem", output)
end

@testset "estimate_size all problem types" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)

    # OnsetProblem (already tested in test_show.jl, but verify here too)
    output = _capture_stdout(() -> estimate_size(OnsetProblem(params)))
    @test occursin("OnsetProblem", output)
    @test occursin("matrix size", output)

    # BiglobalProblem
    cd = ChebyshevDiffn(params.Nr, [params.χ, 1.0], 4)
    bs = conduction_basic_state(cd, params.χ, 6)
    biglobal_output = _capture_stdout(() -> estimate_size(BiglobalProblem(params, bs)))
    @test occursin("BiglobalProblem", biglobal_output)
    @test occursin("matrix size", biglobal_output)

    # TriglobalProblem
    params_tri = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=0, lmax=10, Nr=16)
    bs3d = _basic_state_3d_fixture(params_tri)
    triglobal_output = _capture_stdout(() -> estimate_size(TriglobalProblem(params_tri, bs3d, 0:2)))
    @test occursin("TriglobalProblem", triglobal_output)
    @test occursin("matrix size", triglobal_output)
end

@testset "find_critical_Ra MHDProblem error" begin
    params = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, ricb=0.35,
                       m=1, lmax=6, symm=1, N=8)
    problem = MHDProblem(params)
    @test_throws ErrorException find_critical_Ra(problem)
end

@testset "BiglobalParams show" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    cd = ChebyshevDiffn(params.Nr, [params.χ, 1.0], 4)
    bs = conduction_basic_state(cd, params.χ, 6)
    bp = BiglobalParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16, basic_state=bs)
    output = sprint(show, MIME("text/plain"), bp)
    @test occursin("BiglobalParams", output)
end

@testset "BiglobalParams validation rejects negative Rayleigh" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    cd = ChebyshevDiffn(params.Nr, [params.χ, 1.0], 4)
    bs = conduction_basic_state(cd, params.χ, 6)

    @test_throws ArgumentError BiglobalParams(
        E=1e-3, Pr=1.0, Ra=-100.0, χ=0.35, m=4, lmax=10, Nr=16,
        basic_state=bs)
end
