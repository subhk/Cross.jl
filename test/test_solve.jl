using Test
using Cross
using LinearAlgebra
using Random

@testset "solve(OnsetProblem) integration" begin
    # Use Nr=16 to avoid the "very low" warning (Nr<16 triggers it)
    # and lmax=6 for speed
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=1e5, χ=0.35, m=2, lmax=6, Nr=16)
    problem = OnsetProblem(params)
    result = solve(problem; nev=3)

    @test result isa StabilityResult
    @test length(result.eigenvalues) >= 1  # solver may return fewer than nev
    @test size(result.eigenvectors, 2) == length(result.eigenvalues)
    @test result.growth_rate == maximum(real.(result.eigenvalues))
    @test result.problem === problem

    # Extra should contain operator and info from the onset solver
    @test haskey(result.extra, :operator)

    # Convenience accessors
    @test growth_rate(result) == result.growth_rate
    @test frequency(result) == result.frequency
    @test length(leading_mode(result)) == size(result.eigenvectors, 1)
end

@testset "solve(OnsetProblem) enforces no-slip tau constraints" begin
    Random.seed!(1234)
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=1e5, χ=0.35, m=2, lmax=6, Nr=16,
                         mechanical_bc=:no_slip, thermal_bc=:fixed_temperature)
    problem = OnsetProblem(params)
    result = solve(problem; nev=1, tol=1e-8, maxiter=500)
    op = result.extra.operator
    mode = leading_mode(result)
    D1 = op.cd.D1

    for ell in op.l_sets[:P]
        P_idx = op.index_map[(ell, :P)]
        P = mode[P_idx]
        scale = max(norm(P), eps(Float64))
        @test abs(dot(D1[1, :], P)) / scale < 1e-7
        @test abs(dot(D1[end, :], P)) / scale < 1e-7
    end
end
