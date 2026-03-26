using Test
using Cross

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
