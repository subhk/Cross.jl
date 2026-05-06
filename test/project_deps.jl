using Test
using TOML

@testset "Project dependencies" begin
    project = TOML.parsefile(joinpath(dirname(@__DIR__), "Project.toml"))
    deps = project["deps"]

    @test haskey(deps, "Logging")
end
