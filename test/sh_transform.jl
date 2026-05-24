# Tests for the real-orthonormal SH transform + vector-harmonic horizontal
# divergence (src/BasicStates/sh_transform.jl). These pin the machinery that a
# future correct nonaxisymmetric ū·∇T̄ = ∇·(ūT̄) will use, so it can't regress.

using Test
using Cross
using LinearAlgebra
import Random

@testset "Real-orthonormal SH transform (cos+sin, ±m)" begin
    Random.seed!(11)
    g = Cross.sh_grid(8, 4, Float64)

    @testset "synthesis ↔ analysis round-trip" begin
        c0 = Dict{Tuple{Int,Int},Float64}((ℓ, m) => randn()
              for m in -4:4 for ℓ in abs(m):8)
        c1 = Cross.sh_analyze(Cross.sh_synthesize(c0, g), g)
        err = maximum(abs(c1[k] - c0[k]) for k in keys(c0))
        @test err < 1e-12
    end

    @testset "horizontal divergence: ∇_h·∇_hψ = -ℓ(ℓ+1)ψ" begin
        ψ = Dict{Tuple{Int,Int},Float64}((ℓ, m) => randn()
              for m in -4:4 for ℓ in max(1, abs(m)):6)
        Vθ = Cross.sh_synthesize(ψ, g; Yf=Cross._sh_dYθ)            # ∂θψ
        Vφ = Cross.sh_synthesize(ψ, g; Yf=Cross._sh_dYφ_over_sin)   # (1/sinθ)∂φψ
        div = Cross.sh_horizontal_divergence(Vθ, Vφ, g)
        err = maximum(abs(div[(ℓ, m)] - (-ℓ * (ℓ + 1) * get(ψ, (ℓ, m), 0.0)))
                      for (ℓ, m) in keys(div) if ℓ <= 6)
        @test err < 1e-11
    end

    @testset "toroidal field is divergence-free" begin
        χ = Dict{Tuple{Int,Int},Float64}((ℓ, m) => randn()
              for m in -4:4 for ℓ in max(1, abs(m)):6)
        # u_h = r̂×∇_hχ  ⇒  (Vθ, Vφ) = (-(1/sinθ)∂φχ, ∂θχ)
        Vθ = -Cross.sh_synthesize(χ, g; Yf=Cross._sh_dYφ_over_sin)
        Vφ =  Cross.sh_synthesize(χ, g; Yf=Cross._sh_dYθ)
        div = Cross.sh_horizontal_divergence(Vθ, Vφ, g)
        @test maximum(abs, values(div)) < 1e-12
    end

    @testset "Float32 grid constructs and round-trips" begin
        g32 = Cross.sh_grid(6, 2, Float32)
        c0 = Dict{Tuple{Int,Int},Float32}((ℓ, m) => randn(Float32)
              for m in -2:2 for ℓ in abs(m):6)
        c1 = Cross.sh_analyze(Cross.sh_synthesize(c0, g32), g32)
        @test maximum(abs(c1[k] - c0[k]) for k in keys(c0)) < 1f-4
    end
end
