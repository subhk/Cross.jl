using Test
using LinearAlgebra: I

include("../src/boundary_conditions.jl")

struct DummyOpNoInvR{T}
    Dr::Matrix{T}
    Dθ::Matrix{T}
    Lθ::Matrix{T}
    r::Vector{T}
    sintheta::Vector{T}
    m::Int
end

struct DummyOpWithInvR{T}
    Dr::Matrix{T}
    Dθ::Matrix{T}
    Lθ::Matrix{T}
    r::Vector{T}
    sintheta::Vector{T}
    m::Int
    inv_r::Matrix{T}
end

function build_test_data()
    nr = 4
    ntheta = 3

    Dr = Matrix{ComplexF64}(I, nr, nr)
    Dtheta = Matrix{ComplexF64}(I, ntheta, ntheta)
    Ltheta = 2.0 .* Matrix{ComplexF64}(I, ntheta, ntheta)

    r_desc = ComplexF64[1.0, 0.8, 0.6, 0.4]
    r_asc = reverse(r_desc)
    sin_theta = ComplexF64[0.5, 1.0, 0.5]

    poloidal = reshape(ComplexF64.(collect(1.0:12.0)), nr, ntheta)
    toroidal = reshape(ComplexF64.(collect(13.0:24.0)), nr, ntheta)

    op_desc = DummyOpNoInvR(Dr, Dtheta, Ltheta, r_desc, sin_theta, 2)
    op_asc = DummyOpNoInvR(Dr, Dtheta, Ltheta, r_asc, sin_theta, 2)

    inv_r_matrix = (1.0 ./ r_desc) * ones(ComplexF64, 1, ntheta)
    op_desc_inv = DummyOpWithInvR(Dr, Dtheta, Ltheta, r_desc, sin_theta, 2, inv_r_matrix)

    return (
        nr = nr,
        ntheta = ntheta,
        Dr = Dr,
        r_desc = r_desc,
        r_asc = r_asc,
        poloidal = poloidal,
        toroidal = toroidal,
        op_desc = op_desc,
        op_asc = op_asc,
        op_desc_inv = op_desc_inv,
    )
end

@testset "Mechanical boundary conditions respect r ordering" begin
    data = build_test_data()
    nr = data.nr
    poloidal = data.poloidal
    toroidal = data.toroidal
    Dr = data.Dr

    res_r = fill(ComplexF64(99.0), nr, data.ntheta)
    res_theta = fill(ComplexF64(98.0), nr, data.ntheta)
    res_phi = fill(ComplexF64(97.0), nr, data.ntheta)

    apply_mechanical_bc_from_potentials!(res_r, res_theta, res_phi,
                                         poloidal, toroidal, data.op_desc;
                                         inner = :stress_free,
                                         outer = :no_slip)

    u_r, u_theta, u_phi = velocity_from_potentials(data.op_desc, poloidal, toroidal)
    dr_u_theta = Dr * u_theta
    dr_u_phi = Dr * u_phi
    inv_r = 1.0 ./ data.r_desc

    @test isapprox(res_r[1, :], u_r[1, :])
    @test isapprox(res_theta[1, :], u_theta[1, :])
    @test isapprox(res_phi[1, :], u_phi[1, :])

    @test isapprox(res_r[nr, :], u_r[nr, :])
    @test isapprox(res_theta[nr, :], dr_u_theta[nr, :] .- u_theta[nr, :] .* inv_r[nr])
    @test isapprox(res_phi[nr, :], dr_u_phi[nr, :] .- u_phi[nr, :] .* inv_r[nr])

    @test all(res_r[2:(nr - 1), :] .== ComplexF64(99.0))
    @test all(res_theta[2:(nr - 1), :] .== ComplexF64(98.0))
    @test all(res_phi[2:(nr - 1), :] .== ComplexF64(97.0))
end

@testset "Mechanical BCs with ascending r apply to opposite rows" begin
    data = build_test_data()
    nr = data.nr
    poloidal = data.poloidal
    toroidal = data.toroidal
    Dr = data.Dr

    res_r = fill(ComplexF64(55.0), nr, data.ntheta)
    res_theta = fill(ComplexF64(54.0), nr, data.ntheta)
    res_phi = fill(ComplexF64(53.0), nr, data.ntheta)

    apply_mechanical_bc_from_potentials!(res_r, res_theta, res_phi,
                                         poloidal, toroidal, data.op_asc;
                                         inner = :no_slip,
                                         outer = :stress_free)

    u_r, u_theta, u_phi = velocity_from_potentials(data.op_asc, poloidal, toroidal)
    dr_u_theta = Dr * u_theta
    dr_u_phi = Dr * u_phi
    inv_r = 1.0 ./ data.r_asc

    @test isapprox(res_r[1, :], u_r[1, :])
    @test isapprox(res_theta[1, :], u_theta[1, :])
    @test isapprox(res_phi[1, :], u_phi[1, :])

    @test isapprox(res_r[nr, :], u_r[nr, :])
    @test isapprox(res_theta[nr, :], dr_u_theta[nr, :] .- u_theta[nr, :] .* inv_r[nr])
    @test isapprox(res_phi[nr, :], dr_u_phi[nr, :] .- u_phi[nr, :] .* inv_r[nr])
end

@testset "Mechanical BCs work with matrix inv_r" begin
    data = build_test_data()
    nr = data.nr
    poloidal = data.poloidal
    toroidal = data.toroidal
    Dr = data.Dr

    res_r = fill(ComplexF64(12.0), nr, data.ntheta)
    res_theta = fill(ComplexF64(11.0), nr, data.ntheta)
    res_phi = fill(ComplexF64(10.0), nr, data.ntheta)

    apply_mechanical_bc_from_potentials!(res_r, res_theta, res_phi,
                                         poloidal, toroidal, data.op_desc_inv;
                                         inner = :stress_free,
                                         outer = :no_slip)

    u_r, u_theta, u_phi = velocity_from_potentials(data.op_desc_inv, poloidal, toroidal)
    dr_u_theta = Dr * u_theta
    dr_u_phi = Dr * u_phi
    inv_r = 1.0 ./ data.r_desc

    @test isapprox(res_r[1, :], u_r[1, :])
    @test isapprox(res_theta[1, :], u_theta[1, :])
    @test isapprox(res_phi[1, :], u_phi[1, :])

    @test isapprox(res_theta[nr, :], dr_u_theta[nr, :] .- u_theta[nr, :] .* inv_r[nr])
    @test isapprox(res_phi[nr, :], dr_u_phi[nr, :] .- u_phi[nr, :] .* inv_r[nr])
end

@testset "Thermal boundary conditions respect r ordering" begin
    data = build_test_data()
    nr = data.nr
    Dr = data.Dr

    theta = reshape(ComplexF64.(collect(1.0:12.0)), nr, data.ntheta)
    res_T = fill(ComplexF64(-1.0), nr, data.ntheta)

    apply_thermal_bc_from_potentials!(res_T, theta, data.op_desc;
                                      inner = :fixed_flux,
                                      outer = :fixed_temperature,
                                      value_outer = 3.0,
                                      flux_inner = 2.0)

    dtheta = Dr * theta

    @test isapprox(res_T[1, :], theta[1, :] .- 3.0)
    @test isapprox(res_T[nr, :], dtheta[nr, :] .- 2.0)
end
