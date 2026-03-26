using Test
using Cross
using Logging

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
end

@testset "TriglobalProblem construction and validation" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=0, lmax=10, Nr=16)

    # Build a minimal BasicState3D using the @with_kw keyword constructor
    r_grid = ChebyshevDiffn(16, [0.35, 1.0], 1).x
    bs3d = BasicState3D{Float64}(
        lmax_bs=2, mmax_bs=0, Nr=16,
        r=r_grid,
        theta_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}(),
        dtheta_dr_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}(),
        ur_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}(),
        utheta_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}(),
        uphi_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}(),
        dur_dr_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}(),
        dutheta_dr_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}(),
        duphi_dr_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}()
    )

    problem = TriglobalProblem(params, bs3d, -2:2)
    @test problem.params === params
    @test problem.m_range == -2:2

    # Show works
    output = sprint(show, MIME("text/plain"), problem)
    @test occursin("TriglobalProblem", output)

    # Validation: empty m_range
    @test_throws ArgumentError TriglobalProblem(params, bs3d, 1:0)
end

@testset "MHDProblem construction" begin
    problem = MHDProblem("fake_params")
    @test problem.basic_state === nothing

    output = sprint(show, MIME("text/plain"), problem)
    @test occursin("MHDProblem", output)
end

@testset "estimate_size all problem types" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)

    # OnsetProblem (already tested in test_show.jl, but verify here too)
    output = let buf = IOBuffer()
        redirect_stdout(buf) do
            estimate_size(OnsetProblem(params))
        end
        String(take!(buf))
    end
    @test occursin("OnsetProblem", output)
    @test occursin("Total matrix", output)

    # BiglobalProblem
    cd = ChebyshevDiffn(params.Nr, [params.χ, 1.0], 4)
    bs = conduction_basic_state(cd, params.χ, 6)
    biglobal_output = let buf = IOBuffer()
        redirect_stdout(buf) do
            estimate_size(BiglobalProblem(params, bs))
        end
        String(take!(buf))
    end
    @test occursin("BiglobalProblem", output) || occursin("Total matrix", biglobal_output)

    # TriglobalProblem
    params_tri = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=0, lmax=10, Nr=16)
    r_grid = ChebyshevDiffn(16, [0.35, 1.0], 1).x
    bs3d = BasicState3D{Float64}(
        lmax_bs=2, mmax_bs=0, Nr=16,
        r=r_grid,
        theta_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}(),
        dtheta_dr_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}(),
        ur_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}(),
        utheta_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}(),
        uphi_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}(),
        dur_dr_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}(),
        dutheta_dr_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}(),
        duphi_dr_coeffs=Dict{Tuple{Int,Int},Vector{Float64}}()
    )
    triglobal_output = let buf = IOBuffer()
        redirect_stdout(buf) do
            estimate_size(TriglobalProblem(params_tri, bs3d, 0:2))
        end
        String(take!(buf))
    end
    @test occursin("TriglobalProblem", triglobal_output)
    @test occursin("Total matrix", triglobal_output)
end

@testset "find_critical_Ra MHDProblem error" begin
    problem = MHDProblem("fake_params")
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
