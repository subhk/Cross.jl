using Test
using Cross
using LinearAlgebra

# Assemble the 1-D Galerkin pencil for an A-operator (sum of (power,deriv) terms,
# order q) against an identity mass, with trial recombination R.
function _galerkin_1d(::Type{T}, a_terms, q, R, N, ri, ro) where {T}
    M = N + 1 - q
    A_band = sum(Cross.banded_radial_term(T, p, d, q, N, ri, ro) for (p, d) in a_terms)
    B_band = Cross._convert_up(T, 0, q, N)                  # identity lifted to C^(q)
    A = Cross.galerkin_block(A_band, R, M)
    B = Cross.galerkin_block(B_band, R, M)
    return A, B
end

@testset "banded_radial_term matches sparse_radial_operator on resolved inputs" begin
    T = Float64; ri = 0.35; ro = 1.0; N = 24
    # Validate the r^power multiplication path (multiply in C^(deriv)) against the
    # already-validated sparse_radial_operator. Compare operator ACTION on
    # band-limited inputs (Chebyshev modes up to degree dmax), where r^power·D^deriv
    # stays within the truncation so the two encodings must agree to machine
    # precision. Full-matrix equality fails only in the top truncated rows, which
    # raise polynomial degree past N — and which the Galerkin restriction P_M
    # discards anyway, so they never enter the assembled pencil.
    for (power, deriv) in [(0,1),(1,0),(2,0),(2,1),(2,2),(3,1),(4,2),(1,1)]
        dmax = N - power - deriv
        banded = Matrix(Cross.banded_radial_term(T, power, deriv, deriv, N, ri, ro))  # C^(deriv)
        lifted = Matrix(Cross._convert_up(T, 0, deriv, N) *
                        Cross.sparse_radial_operator(power, deriv, N, ri, ro))        # C^0 -> C^(deriv)
        @test isapprox(banded[:, 1:dmax+1], lifted[:, 1:dmax+1]; atol=1e-8, rtol=1e-8)
    end
end

@testset "Galerkin 1-D Dirichlet Laplacian: analytic spectrum, no spurious" begin
    T = Float64; ri = 0.35; ro = 1.0; L = ro - ri
    for N in (24, 32, 48)
        R = Cross.recomb_dirichlet(T, N)
        A, B = _galerkin_1d(T, [(0, 2)], 2, R, N, ri, ro)   # u'' = λ u
        vals = sort(filter(isfinite, real.(eigen(A, B).values)))
        @test maximum(vals) < 1e-6                          # no positive spurious
        analytic = [-(n * π / L)^2 for n in 1:5]
        got = sort(sort(vals; by = abs)[1:5])
        @test isapprox(got, sort(analytic); rtol = 1e-6)
    end
end

@testset "Galerkin 1-D Neumann Laplacian: analytic spectrum" begin
    T = Float64; ri = 0.35; ro = 1.0; L = ro - ri
    N = 40
    R = Cross.recomb_neumann(T, N)
    A, B = _galerkin_1d(T, [(0, 2)], 2, R, N, ri, ro)
    vals = sort(filter(isfinite, real.(eigen(A, B).values)))
    @test maximum(vals) < 1e-6
    got = sort(vals; by = abs)
    @test isapprox(got[1], 0.0; atol = 1e-6)                # constant Neumann mode
    @test isapprox(sort(got[2:6]), sort([-(n*π/L)^2 for n in 1:5]); rtol = 1e-6)
end

@testset "Galerkin 1-D clamped beam: analytic spectrum, no spurious" begin
    T = Float64; ri = 0.35; ro = 1.0; L = ro - ri
    β = (4.730040744862704, 7.853204624095838, 10.995607838001671)
    for N in (32, 48)
        R = Cross.recomb_clamped(T, N)
        A, B = _galerkin_1d(T, [(0, 4)], 4, R, N, ri, ro)   # u'''' = λ u
        vals = sort(filter(isfinite, real.(eigen(A, B).values)))
        @test minimum(vals) > -1e-3                          # positive definite
        got = sort(sort(vals; by = abs)[1:3])
        @test isapprox(got, sort([(b/L)^4 for b in β]); rtol = 1e-5)
    end
end

@testset "Stress-free recombinations: nullity + BCs (independent of library helpers)" begin
    T = Float64; ri = 0.35; ro = 1.0; N = 32; K = N + 1
    scale = Cross._radial_scale(ri, ro)
    Rp = Cross.recomb_poloidal_stressfree(T, N, ri, ro)
    Rt = Cross.recomb_toroidal_stressfree(T, N, ri, ro)
    @test size(Rp) == (K, N - 3)        # q=4 (u=0, r·u''=0 both ends)
    @test size(Rt) == (K, N - 1)        # q=2 (-r·v'+v=0 both ends)

    # First-principles boundary evaluators (no Cross._chebyshev_boundary_* helpers):
    #   T_n(1)=1, T_n(-1)=(-1)^n ; T_n'(1)=n^2, T_n'(-1)=(-1)^(n+1) n^2 ;
    #   T_n''(1)=n^2(n^2-1)/3, T_n''(-1)=(-1)^n n^2(n^2-1)/3
    val_o = T[1 for n in 0:N];            val_i = T[(-1)^n for n in 0:N]
    der_o = T[n^2 for n in 0:N];          der_i = T[(-1)^(n+1) * n^2 for n in 0:N]
    sec_o = T[n^2*(n^2-1)/3 for n in 0:N]; sec_i = T[(-1)^n * n^2*(n^2-1)/3 for n in 0:N]
    rb_o = Cross._boundary_radius(ri, ro, :outer)
    rb_i = Cross._boundary_radius(ri, ro, :inner)

    # Poloidal stress-free: u=0 and u''=0 at both boundaries.
    @test maximum(abs.(val_o' * Rp)) < 1e-8
    @test maximum(abs.(val_i' * Rp)) < 1e-8
    @test maximum(abs.((scale^2 .* sec_o)' * Rp)) < 1e-6
    @test maximum(abs.((scale^2 .* sec_i)' * Rp)) < 1e-6

    # Toroidal stress-free: -r·v' + v = 0 at both boundaries.
    @test maximum(abs.((@. -rb_o * scale * der_o + val_o)' * Rt)) < 1e-8
    @test maximum(abs.((@. -rb_i * scale * der_i + val_i)' * Rt)) < 1e-8
    # ...and it must NOT force v=0 (distinguishes the Robin basis from Dirichlet).
    @test maximum(abs.(val_o' * Rt)) > 1e-3
end
