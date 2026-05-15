using Test
using SparseArrays
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

    cache = Cross._build_azimuthal_coupling_cache(1, 4, 4, T)
    @test typeof(cache.weight) === T
    @test eltype(cache.y_m) === T
    @test eltype(cache.y_0) === T

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

@testset "Critical-Rayleigh helpers accept Float32 inputs" begin
    T = Float32
    failing_builder = Ra -> error("synthetic failure")
    failing_factory = (E, χ, Pr, m) -> error("synthetic failure")

    @test_throws ErrorException Cross.find_critical_rayleigh(
        failing_builder, T(1e-3), T(0.35), 1;
        Ra_min = T(1), Ra_max = T(2), tol = T(1e-3), growth_tol = T(1e-3))

    _, _, _, results = Cross.find_onset_parameters(
        failing_factory, T(1e-3), T(0.35), one(T), [1])

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
    @test eltype(sparse_op.r0_D0_u) === T
    @test eltype(sparse_op.r2_D2_u) === T
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
    @test eltype(mhd_op.r0_D0_u) === T
    @test eltype(mhd_op.r0_D0_f) === T
    @test valtype(typeof(mhd_op.background_ops)) === SparseMatrixCSC{T, Int}
    @test eltype(Cross.sparse_background_operator(4, 0, 4, mhd_params)) === T

    induction_block = Cross.operator_induction_poloidal_from_v(mhd_op, 2, 1, -1)
    lorentz_block = Cross.operator_lorentz_poloidal_offdiag(mhd_op, 2, 1, -1, mhd_params.Le)
    @test eltype(induction_block) === ComplexF32
    @test eltype(lorentz_block) === ComplexF32

    A_mhd, B_mhd, _, _ = assemble_mhd_matrices(mhd_op)

    @test eltype(A_mhd) === ComplexF32
    @test eltype(B_mhd) === ComplexF32

    axial_params = MHDParams(
        E = T(1e-3),
        Pr = one(T),
        Pm = one(T),
        Ra = T(100),
        Le = one(T),
        ricb = T(0.35),
        m = 1,
        lmax = 3,
        N = 8,
        B0_type = axial,
        B0_amplitude = one(T)
    )
    axial_op = MHDStabilityOperator(axial_params)
    axial_btor_block = Cross.operator_lorentz_poloidal_offdiag(
        axial_op, 2, 1, -1, axial_params.Le)
    axial_bpol_block = Cross.operator_lorentz_poloidal_from_bpol(
        axial_op, 3, 1, -2, axial_params.Le)
    @test eltype(axial_btor_block) === ComplexF32
    @test eltype(axial_bpol_block) === ComplexF32
end

@testset "Public MHD solve preserves Float32 result storage" begin
    T = Float32
    params = MHDParams(
        E = T(1e-3),
        Pr = one(T),
        Pm = one(T),
        Ra = T(100),
        Le = one(T),
        ricb = T(0.35),
        m = 1,
        lmax = 3,
        N = 8,
        B0_type = dipole,
        B0_amplitude = one(T)
    )

    result = solve(MHDProblem(params); nev=1, sigma=zero(T), maxiter=20)

    @test eltype(result.eigenvalues) === ComplexF32
    @test eltype(result.eigenvectors) === ComplexF32
end

@testset "Sparse eigensolver preserves Float32 storage" begin
    A = spdiagm(0 => ComplexF32[1, 2, 3, 4, 5, 6])
    B = spdiagm(0 => ComplexF32[1, 1, 1, 1, 1, 1])

    eigenvalues, eigenvectors, _ = Cross.solve_eigenvalue_problem(
        A, B; nev=1, sigma=0.0f0, krylovdim=4, maxiter=20, verbosity=0)

    @test eltype(eigenvalues) === ComplexF32
    @test eltype(eigenvectors) === ComplexF32
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

    empty_coeffs = Dict{Int, Vector{ComplexF32}}()
    empty_field = Cross.spectral_to_physical(empty_coeffs, grid, params.Nr)
    @test eltype(empty_field) === ComplexF32

    r = Float32.(range(0.35, 1; length=8))
    θ = Float32.(range(0.1, 3.0; length=10))
    ur_grid = randn(ComplexF32, length(r), length(θ))
    uθ_grid = randn(ComplexF32, length(r), length(θ))
    ψ = Cross.meridional_streamfunction(ur_grid, uθ_grid, r, θ, 0)
    @test eltype(ψ) === ComplexF32
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
