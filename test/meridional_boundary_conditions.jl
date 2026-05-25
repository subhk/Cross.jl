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
        (1, 1) => 0.2 .+ r .* (1 .- r),
        (2, 1) => 0.1 .* r.^2,
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

@testset "Meridional sin partner is the φ-rotation of the cos mode" begin
    # A sin temperature mode (ℓ0,-m0) is the φ-rotation (by π/2m0) of the cos mode
    # (ℓ0,+m0), so it must drive a meridional flow with the SAME radial profile,
    # stored under the -m0 key. (Pre-fix the solver looped m_bs in 0:mmax and the
    # sin partner drove zero flow.) Also pins that the cos (m≥0) path is unchanged.
    χ, E, Ra, Pr = 0.35, 1e-3, 2e4, 1.0
    Nr, lmax, mmax, ℓ0, m0 = 24, 6, 2, 2, 2
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 2)
    r = cd.x; D1 = Matrix(cd.D1); D2 = Matrix(cd.D2)
    prof = sin.(π .* (r .- χ) ./ (1 - χ))

    function run_merid(theta_key)
        theta = Dict{Tuple{Int,Int},Vector{Float64}}(theta_key => copy(prof))
        ur  = Dict{Tuple{Int,Int},Vector{Float64}}()
        uth = Dict{Tuple{Int,Int},Vector{Float64}}()
        dur = Dict{Tuple{Int,Int},Vector{Float64}}()
        dth = Dict{Tuple{Int,Int},Vector{Float64}}()
        solve_meridional_circulation_toroidal_poloidal!(
            ur, uth, dur, dth, theta, Dict{Tuple{Int,Int},Vector{Float64}}(),
            r, D1, D2, χ, 1.0, Ra, E, Pr, lmax, mmax; mechanical_bc=:no_slip)
        return ur, uth
    end

    ur_c, uth_c = run_merid((ℓ0,  m0))   # cos temperature
    ur_s, uth_s = run_merid((ℓ0, -m0))   # sin temperature (rotated)

    # cos mode drives a nonzero flow
    @test maximum(maximum(abs, v) for v in values(uth_c)) > 1e-6
    # sin mode now drives a flow of equal magnitude under the -m0 keys
    @test maximum((maximum(abs, get(uth_s, (ℓ, -m0), [0.0])) for ℓ in m0:lmax); init=0.0) > 1e-6
    # exact rotation invariance: sin(-m0) profile == cos(+m0) profile
    for ℓ in m0:lmax
        @test get(uth_c, (ℓ, m0), zeros(Nr)) ≈ get(uth_s, (ℓ, -m0), zeros(Nr)) atol=1e-12
        @test get(ur_c,  (ℓ, m0), zeros(Nr)) ≈ get(ur_s,  (ℓ, -m0), zeros(Nr)) atol=1e-12
    end
end
