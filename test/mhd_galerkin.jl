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

@testset "solve(MHDProblem) hydro path: spurious-free, matches onset end-to-end" begin
    E = 4.225e-4; Pr = 1.0; χ = 0.35; m = 4; Ra = 55.905; lmax = 8; Nr = 32
    onset_lead = maximum(real.(solve(OnsetProblem(OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ,
                                     m=m, lmax=lmax, Nr=Nr)); nev=12).eigenvalues))
    res = solve(MHDProblem(MHDParams(E=E, Pr=Pr, Ra=Ra, ricb=χ, m=m, lmax=lmax, N=Nr, symm=0));
                nev=8, which=:LR)
    # Public solve(): :LR selects the convective mode with no σ-targeting, matching onset.
    @test isapprox(maximum(real.(res.eigenvalues)), onset_lead; atol=5e-3)
    @test maximum(real.(res.eigenvalues)) < 0.1                  # no +real spurious returned
    @test size(res.eigenvectors, 2) == length(res.eigenvalues)   # structural
    @test size(res.eigenvectors, 1) > 0                          # full-size eigenvectors reconstructed
end

@testset "assemble_mhd_galerkin guards the (reverted) magnetic sector" begin
    # Magnetic Galerkin coupling was reverted (G3.2b); magnetic cases route through
    # the tau path, so assembly errors on a field.
    op = Cross.MHDStabilityOperator(MHDParams(E=1e-3, Pr=1.0, Ra=100.0, ricb=0.35, m=4,
                                              lmax=6, N=16, B0_type=Cross.axial, Le=0.1))
    @test_throws ErrorException Cross.assemble_mhd_galerkin(op)
end

@testset "Magnetic diffusion sign: free-decay modes dissipate (≡ viscous at Pm=1)" begin
    # Magnetic diffusion enters A with a MINUS sign (assembly.jl), like the identical
    # viscous operator. Decoupled free-decay must dissipate (Re<0); the old +sign made
    # it grow. This pins the corrected sign. (Cross-check vs Kore still advised.)
    T = Float64; E = 1e-3; Pm = 1.0; Em = E / Pm; m = 2; lmax = 4; Nr = 16; χ = 0.35; N = Nr; ℓ = 2
    op = Cross.MHDStabilityOperator(MHDParams(E=E, Pr=1.0, Pm=Pm, Ra=1e4, ricb=χ, m=m,
                                              lmax=lmax, N=Nr, symm=0, B0_type=Cross.axial,
                                              Le=1e-2, B0_amplitude=1.0))
    Rd = Cross.recomb_dirichlet(T, N); Md = N - 1
    lift(blk) = Cross.galerkin_block(Cross._convert_up(T, 0, 2, N) * blk, Rd, Md)
    decay(A0, B0) = sort(real.(filter(isfinite, eigen(lift(A0), lift(B0)).values)); rev=true)
    mag  = decay(-Cross.operator_magnetic_diffusion_toroidal(op, ℓ, Em), -Cross.operator_b_toroidal(op, ℓ))
    visc = decay(-Cross.operator_viscous_toroidal(op, ℓ, E),            -Cross.operator_u_toroidal(op, ℓ))
    grow = decay(+Cross.operator_magnetic_diffusion_toroidal(op, ℓ, Em), -Cross.operator_b_toroidal(op, ℓ))
    @test maximum(mag) < 0                          # corrected sign ⇒ dissipates
    @test isapprox(mag, visc; rtol=1e-8)            # at Pm=1, magnetic ≡ viscous free-decay
    @test maximum(grow) > 1.0                       # the old +sign grew (documents the bug)
end
