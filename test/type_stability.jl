using Test
using Cross

@testset "Public wrapper type stability" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
    problem = OnsetProblem(params)
    op = LinearStabilityOperator(params)

    get_params(x) = x.params

    @test isconcretetype(fieldtype(typeof(problem), :params))
    @test isconcretetype(fieldtype(typeof(op), :params))
    @inferred get_params(problem)
    @inferred LinearStabilityOperator(params)
    @inferred assemble_matrices(op)
    @inferred Cross._check_memory(problem, "OnsetProblem")
end

@testset "Leading mode avoids vector copy" begin
    eigenvalues = [complex(0.1, 2.0), complex(0.5, -1.0), complex(-0.2, 0.3)]
    eigenvectors = hcat([1.0+0im, 0, 0], [0, 1.0+0im, 0], [0, 0, 1.0+0im])
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
    problem = OnsetProblem(params)
    result = StabilityResult(eigenvalues, eigenvectors, problem)

    mode = leading_mode(result)
    @test mode == eigenvectors[:, 2]
    @test Base.mightalias(mode, result.eigenvectors)
end
