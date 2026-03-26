using Test
using Cross

@testset "OnsetParams show" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=30, Nr=64)
    output = sprint(show, MIME("text/plain"), params)

    @test occursin("OnsetParams", output)
    @test occursin("no_slip", output)
    @test occursin("fixed_temperature", output)
end

@testset "StabilityResult show" begin
    eigenvalues = [complex(0.1, 2.0), complex(0.5, -1.0)]
    eigenvectors = rand(ComplexF64, 10, 2)
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    problem = OnsetProblem(params)
    result = StabilityResult(eigenvalues, eigenvectors, problem)
    output = sprint(show, MIME("text/plain"), result)

    @test occursin("StabilityResult", output)
    @test occursin("2 eigenvalues", output)
    @test occursin("Growth rate", output)
end

@testset "OnsetProblem show" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    problem = OnsetProblem(params)
    output = sprint(show, MIME("text/plain"), problem)

    @test occursin("OnsetProblem", output)
end

@testset "estimate_size runs without error" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    problem = OnsetProblem(params)

    output = let buf = IOBuffer()
        redirect_stdout(buf) do
            estimate_size(problem)
        end
        String(take!(buf))
    end
    @test occursin("OnsetProblem", output)
    @test occursin("Total matrix", output)
end
