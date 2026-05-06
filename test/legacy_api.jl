using Test
using TOML
using Cross

@testset "Legacy v1 API is removed" begin
    @test !isdefined(Cross, :ShellParams)
    @test !isdefined(Cross, :leading_modes)
    @test !isdefined(Cross, :_arnoldi_eigensolve)

    project = TOML.parsefile(joinpath(dirname(@__DIR__), "Project.toml"))
    @test !haskey(project["deps"], "ArnoldiMethod")
    @test !haskey(project["compat"], "ArnoldiMethod")
end
