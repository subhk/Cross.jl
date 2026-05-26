using Test
using Cross
using LinearAlgebra

@testset "MHD(B0→0) Galerkin hydro reduces to collocation onset (no spurious)" begin
    E = 4.225e-4; Pr = 1.0; χ = 0.35; m = 4; Ra = 55.905; lmax = 8; Nr = 32

    # Reference: validated collocation onset solver at the benchmark config.
    onset = solve(OnsetProblem(OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=m, lmax=lmax, Nr=Nr));
                  nev=12, which=:LR)
    onset_lead = maximum(real.(onset.eigenvalues))

    # MHD hydro (no background field), tau-free Galerkin assembly.
    mhd_params = MHDParams(E=E, Pr=Pr, Ra=Ra, ricb=χ, m=m, lmax=lmax, N=Nr, symm=0)
    op = Cross.MHDStabilityOperator(mhd_params)
    A, B, layout = Cross.assemble_mhd_galerkin(op)

    ev = filter(isfinite, eigen(Matrix(A), Matrix(B)).values)
    gal_lead = maximum(real.(ev))

    @test op.ll_u == op.ll_h                          # temperature tracks poloidal
    @test layout.nred == length(op.ll_u)*(Nr-3) + (length(op.ll_v)+length(op.ll_h))*(Nr-1)
    @test length(ev) == layout.nred                    # full-rank B (no infinite eigenvalues)
    @test count(>(0.1), real.(ev)) == 0                # NO +real spurious (the MHD pathology)
    # :maxreal lands on the convective mode with no σ-targeting, matching onset.
    @test isapprox(gal_lead, onset_lead; atol=5e-3)
end
