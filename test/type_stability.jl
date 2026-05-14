using Test
using Cross

@testset "Public wrapper type stability" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
    problem = OnsetProblem(params)
    op = LinearStabilityOperator(params)

    get_params(x) = x.params

    @test isconcretetype(fieldtype(typeof(problem), :params))
    @test isconcretetype(fieldtype(typeof(op), :params))
    @inferred get_params(problem)
    @inferred LinearStabilityOperator(params)
    @inferred assemble_matrices(op)
    @inferred Cross._check_memory(problem, "OnsetProblem")
end

@testset "Leading mode avoids vector copy" begin
    eigenvalues = [complex(0.1, 2.0), complex(0.5, -1.0), complex(-0.2, 0.3)]
    eigenvectors = hcat([1.0+0im, 0, 0], [0, 1.0+0im, 0], [0, 0, 1.0+0im])
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
    problem = OnsetProblem(params)
    result = StabilityResult(eigenvalues, eigenvectors, problem)

    mode = leading_mode(result)
    @test mode == eigenvectors[:, 2]
    @test Base.mightalias(mode, result.eigenvectors)
end

@testset "Sparse radial operator avoids dense RHS materialization" begin
    Cross.sparse_radial_operator(4, 4, 256, 0.35, 1.0)
    GC.gc()
    bytes = @allocated Cross.sparse_radial_operator(4, 4, 256, 0.35, 1.0)

    @test bytes < 4_800_000
end

@testset "Constraint reduction avoids dense subblock copies" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=10, Nr=24)
    op = LinearStabilityOperator(params)
    A, B, interior_dofs, boundary_dofs = assemble_matrices(op)

    Cross._constrained_reduced_matrices(A, B, op, interior_dofs, boundary_dofs)
    GC.gc()
    bytes = @allocated Cross._constrained_reduced_matrices(
        A, B, op, interior_dofs, boundary_dofs)

    @test bytes < 30_000_000
end

@testset "MHD background operator avoids dense RHS materialization" begin
    params = MHDParams(
        E = 1e-3,
        Pr = 1.0,
        Pm = 1.0,
        Ra = 100.0,
        Le = 1.0,
        ricb = 0.35,
        m = 1,
        lmax = 4,
        N = 256,
        B0_type = dipole,
        B0_amplitude = 1.0
    )

    Cross.sparse_background_operator(4, 0, 4, params)
    GC.gc()
    bytes = @allocated Cross.sparse_background_operator(4, 0, 4, params)

    @test bytes < 9_300_000
end

@testset "Basic state operator blocks preserve real precision" begin
    T = Float32
    Nr = 12
    χ = T(0.35)
    cd = ChebyshevDiffn(Nr, T[χ, one(T)], 4)
    r = cd.x
    theta = fill(T(0.1), Nr)
    uphi = fill(T(0.2), Nr)
    zero_coeff = zeros(T, Nr)
    bs = BasicState{T}(
        lmax_bs = 1,
        Nr = Nr,
        r = r,
        theta_coeffs = Dict(0 => zero_coeff, 1 => theta),
        uphi_coeffs = Dict(0 => zero_coeff, 1 => uphi),
        dtheta_dr_coeffs = Dict(0 => zero_coeff, 1 => cd.D1 * theta),
        duphi_dr_coeffs = Dict(0 => zero_coeff, 1 => cd.D1 * uphi)
    )
    params = OnsetParams(
        E = T(1e-3),
        Pr = one(T),
        Ra = T(100),
        χ = χ,
        m = 1,
        lmax = 4,
        Nr = Nr,
        basic_state = bs
    )
    op = LinearStabilityOperator(params)
    bs_ops = Cross.build_basic_state_operators(bs, op, params.m)
    block_dicts = (
        bs_ops.advection_blocks,
        bs_ops.shear_radial_blocks,
        bs_ops.shear_theta_blocks,
        bs_ops.shear_theta_toroidal_blocks,
        bs_ops.temp_grad_radial_blocks,
        bs_ops.temp_grad_theta_blocks,
        bs_ops.temp_grad_theta_toroidal_blocks,
        bs_ops.metric_poloidal_blocks
    )
    blocks = [block for dict in block_dicts for block in values(dict)]

    @test !isempty(blocks)
    @test all(eltype(block) == ComplexF32 for block in blocks)

    summary = analyze_basic_state(bs; verbose=false)
    @test valtype(typeof(summary)) === NamedTuple{(:θ_max, :uphi_max), Tuple{T, T}}
end

@testset "Onset scan result dictionary keeps integer keys" begin
    failing_factory = (E, χ, Pr, m) -> error("synthetic failure")

    _, _, _, results = Cross.find_onset_parameters(
        failing_factory, 1e-3, 0.35, 1.0, [1, 2])

    @test keytype(typeof(results)) === Int
    @test valtype(typeof(results)) !== Any
end

@testset "Sparse assemblies preserve Float32 storage" begin
    T = Float32
    sparse_params = SparseOnsetParams(
        E = T(1e-3),
        Pr = one(T),
        Ra = T(100),
        ricb = T(0.35),
        m = 1,
        lmax = 4,
        N = 16
    )
    sparse_op = SparseStabilityOperator(sparse_params)
    A_sparse, B_sparse, _, _ = assemble_sparse_matrices(sparse_op)

    @test eltype(A_sparse) === ComplexF32
    @test eltype(B_sparse) === ComplexF32

    mhd_params = MHDParams(
        E = T(1e-3),
        Pr = one(T),
        Pm = one(T),
        Ra = T(100),
        Le = one(T),
        ricb = T(0.35),
        m = 1,
        lmax = 3,
        N = 16,
        B0_type = dipole,
        B0_amplitude = one(T)
    )
    mhd_op = MHDStabilityOperator(mhd_params)
    A_mhd, B_mhd, _, _ = assemble_mhd_matrices(mhd_op)

    @test eltype(A_mhd) === ComplexF32
    @test eltype(B_mhd) === ComplexF32
end

@testset "Velocity reconstruction preserves Float32 precision and avoids synthesis temporaries" begin
    T = Float32
    params = OnsetParams(
        E = T(1e-3),
        Pr = one(T),
        Ra = T(100),
        χ = T(0.35),
        m = 2,
        lmax = 12,
        Nr = 32
    )
    op = LinearStabilityOperator(params)
    eigenvector = randn(ComplexF32, op.total_dof)

    ur, uθ, uφ, grid = Cross.eigenvector_to_velocity(eigenvector, op)
    @test eltype(ur) === ComplexF32
    @test eltype(uθ) === ComplexF32
    @test eltype(uφ) === ComplexF32
    @test grid isa Cross.MeridionalGrid{T}

    P_coeffs = Dict(ℓ => randn(ComplexF32, params.Nr) for ℓ in op.l_sets[:P])
    Cross.spectral_to_physical(P_coeffs, grid, params.Nr)
    GC.gc()
    bytes = @allocated Cross.spectral_to_physical(P_coeffs, grid, params.Nr)

    @test bytes < 100_000
end

@testset "Axisymmetric basic-state extraction avoids eager zero defaults" begin
    T = Float64
    Nr = 48
    lmax = 64
    r = collect(range(T(0.35), one(T), length=Nr))
    zero_coeff = zeros(T, Nr)
    coeffs = Dict{Tuple{Int,Int}, Vector{T}}((ℓ, 0) => zero_coeff for ℓ in 0:lmax)
    empty = Dict{Tuple{Int,Int}, Vector{T}}()
    bs3d = BasicState3D{T}(
        lmax_bs = lmax,
        mmax_bs = 0,
        Nr = Nr,
        r = r,
        theta_coeffs = coeffs,
        dtheta_dr_coeffs = copy(coeffs),
        ur_coeffs = empty,
        utheta_coeffs = empty,
        uphi_coeffs = copy(coeffs),
        dur_dr_coeffs = empty,
        dutheta_dr_coeffs = empty,
        duphi_dr_coeffs = copy(coeffs)
    )

    Cross.axisymmetric_basic_state(bs3d)
    GC.gc()
    bytes = @allocated Cross.axisymmetric_basic_state(bs3d)

    @test bytes < 80_000
end
