using Test
using LinearAlgebra
using Cross

function _stress_free_residual(D1, r, u, idx)
    return dot(D1[idx, :], u) - u[idx] / r[idx]
end

@testset "Meridional circulation enforces stress-free BCs" begin
    cd = ChebyshevDiffn(10, [0.35, 1.0], 2)
    r = cd.x
    D1 = cd.D1
    D2 = cd.D2
    idx_inner = 1
    idx_outer = length(r)

    theta_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}(
        (1, 1) => @. 0.2 + r * (1 - r),
        (2, 1) => @. 0.1 * r^2,
    )
    uphi_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}()
    ur_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}()
    utheta_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}()
    dur_dr_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}()
    dutheta_dr_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}()

    solve_meridional_circulation_toroidal_poloidal!(
        ur_coeffs, utheta_coeffs, dur_dr_coeffs, dutheta_dr_coeffs,
        theta_coeffs, uphi_coeffs,
        r, D1, D2, first(r), last(r),
        1e5, 1e-3, 1.0, 2, 1;
        mechanical_bc = :stress_free,
        include_meridional = true,
        use_full_coupling = true,
    )

    for ell in 1:2
        uθ = utheta_coeffs[(ell, 1)]
        scale = max(norm(uθ), eps(Float64))
        @test abs(_stress_free_residual(D1, r, uθ, idx_inner)) / scale < 1e-9
        @test abs(_stress_free_residual(D1, r, uθ, idx_outer)) / scale < 1e-9
    end
end
