using Test
using Cross

@testset "Legacy banner API is removed" begin
    @test !isdefined(Cross, :CROSS_BANNER)
    @test !isdefined(Cross, :print_cross_header)
end

@testset "OnsetParams show" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=30, Nr=64)
    output = sprint(show, MIME("text/plain"), params)

    @test startswith(output, "OnsetParams{Float64}")
    @test occursin("├── dynamics: E=0.001, Pr=1.0, Ra=100.0", output)
    @test occursin("├── geometry: χ=0.35", output)
    @test occursin("├── resolution: m=4, lmax=30, Nr=64", output)
    @test occursin("├── boundary conditions: mechanical=no_slip, thermal=fixed_temperature", output)
    @test occursin("└── equatorial symmetry: both", output)
end

@testset "StabilityResult show" begin
    eigenvalues = [complex(0.1, 2.0), complex(0.5, -1.0)]
    eigenvectors = rand(ComplexF64, 10, 2)
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    problem = OnsetProblem(params)
    result = StabilityResult(eigenvalues, eigenvectors, problem)
    output = sprint(show, MIME("text/plain"), result)

    @test startswith(output, "StabilityResult{Float64} with 2 eigenvalues")
    @test occursin("├── leading eigenvalue: 0.5 - 1.0im", output)
    @test occursin("├── growth rate: 0.5", output)
    @test occursin("├── frequency: -1.0", output)
    @test occursin("└── problem: OnsetProblem", output)
end

@testset "OnsetProblem show" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    problem = OnsetProblem(params)
    output = sprint(show, MIME("text/plain"), problem)

    @test startswith(output, "OnsetProblem{Float64}")
    @test occursin("├── parameters: E=0.001, Ra=100.0, Pr=1.0, χ=0.35", output)
    @test occursin("├── resolution: m=4, lmax=10, Nr=16", output)
    @test occursin("└── boundary conditions: mechanical=no_slip, thermal=fixed_temperature", output)
end

@testset "estimate_size uses tree summary" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    problem = OnsetProblem(params)

    output = mktemp() do _, io
        redirect_stdout(io) do
            estimate_size(problem)
        end
        flush(io)
        seekstart(io)
        read(io, String)
    end
    @test startswith(output, "OnsetProblem size estimate")
    @test occursin("├── l-modes:", output)
    @test occursin("├── degrees of freedom per mode:", output)
    @test occursin("├── matrix size:", output)
    @test occursin("└── dense storage estimate:", output)
end
