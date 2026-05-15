using Test
using LinearAlgebra
using SparseArrays
using Logging
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

@testset "Dense hydrodynamic solver accepts Float32 operators" begin
    T = Float32
    params = OnsetParams(
        E = T(1e-3),
        Pr = one(T),
        Ra = T(100),
        χ = T(0.35),
        m = 1,
        lmax = 3,
        Nr = 8
    )

    result_or_error = try
        solve(OnsetProblem(params); nev=1, sigma=zero(T), maxiter=20)
    catch err
        err
    end

    @test !(result_or_error isa Exception)
    if !(result_or_error isa Exception)
        @test eltype(result_or_error.eigenvalues) === ComplexF32
        @test eltype(result_or_error.eigenvectors) === ComplexF32
    end
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

    @inferred Cross.eigenvector_to_velocity(eigenvector, op)
    @inferred Cross.eigenvector_to_velocity(eigenvector, op; grid=grid)

    Nr_alloc = 64
    Nθ_alloc = 128
    cd_alloc = ChebyshevDiffn(Nr_alloc, Float64[0.35, 1.0], 1)
    grid_alloc = Cross.build_meridional_grid(Nθ_alloc, 2, 16; T=Float64)
    P_alloc = randn(ComplexF64, Nr_alloc, Nθ_alloc)
    T_alloc = randn(ComplexF64, Nr_alloc, Nθ_alloc)
    Cross.potentials_to_velocity(P_alloc, T_alloc;
                                 Dr=cd_alloc.D1,
                                 Dθ=grid_alloc.Dθ,
                                 Lθ=grid_alloc.Lθ,
                                 r=cd_alloc.x,
                                 sintheta=grid_alloc.sinθ,
                                 m=2)
    GC.gc()
    velocity_bytes = @allocated Cross.potentials_to_velocity(
        P_alloc, T_alloc;
        Dr=cd_alloc.D1,
        Dθ=grid_alloc.Dθ,
        Lθ=grid_alloc.Lθ,
        r=cd_alloc.x,
        sintheta=grid_alloc.sinθ,
        m=2)
    @test velocity_bytes < 700_000

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

@testset "Triglobal reconstruction helpers preserve Float32 grids" begin
    T = Float32
    typed_grid = try
        Cross._build_chebyshev_grid(8, T(0.35), one(T))
    catch err
        err
    end
    mixed_grid = try
        Cross._build_chebyshev_grid(8, T(0.35), 1.0)
    catch err
        err
    end

    @test !(typed_grid isa Exception)
    @test !(mixed_grid isa Exception)
    if !(typed_grid isa Exception)
        @test eltype(typed_grid.x) === T
        @test eltype(typed_grid.D1) === T
    end
    if !(mixed_grid isa Exception)
        @test eltype(mixed_grid.x) === T
        @test eltype(mixed_grid.D1) === T
    end
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

@testset "Typed sparse helpers avoid Float64 empty and boundary temporaries" begin
    T = Float32
    empty_terms = Tuple{T, SparseMatrixCSC{T, Int}}[]
    empty_block = Cross.combine_terms(empty_terms)

    @test eltype(empty_block) === T

    boundary_values = try
        Cross._chebyshev_boundary_values(8, :outer, T)
    catch err
        err
    end
    boundary_derivative = try
        Cross._chebyshev_boundary_derivative(8, :inner, T)
    catch err
        err
    end
    boundary_second = try
        Cross._chebyshev_boundary_second_derivative(8, :inner, T)
    catch err
        err
    end

    @test !(boundary_values isa Exception)
    @test !(boundary_derivative isa Exception)
    @test !(boundary_second isa Exception)
    if !(boundary_values isa Exception)
        @test eltype(boundary_values) === T
    end
    if !(boundary_derivative isa Exception)
        @test eltype(boundary_derivative) === T
    end
    if !(boundary_second isa Exception)
        @test eltype(boundary_second) === T
    end
end

@testset "Thermal wind balance reuses mode-independent dense operator" begin
    T = Float64
    Nr = 48
    cd = ChebyshevDiffn(Nr, T[0.35, 1.0], 2)
    theta_coeffs = Dict{Int, Vector{T}}(
        ℓ => fill(T(0.05) / T(ℓ + 1), Nr) for ℓ in 1:18)

    uphi_coeffs = Dict{Int, Vector{T}}()
    duphi_dr_coeffs = Dict{Int, Vector{T}}()
    Cross.solve_thermal_wind_balance!(
        uphi_coeffs, duphi_dr_coeffs, theta_coeffs, cd, T(0.35), one(T), T(100), one(T))

    GC.gc()
    uphi_coeffs = Dict{Int, Vector{T}}()
    duphi_dr_coeffs = Dict{Int, Vector{T}}()
    bytes = @allocated Cross.solve_thermal_wind_balance!(
        uphi_coeffs, duphi_dr_coeffs, theta_coeffs, cd, T(0.35), one(T), T(100), one(T))

    @test bytes < 1_000_000
end

@testset "Coupled thermal wind assembly avoids dense block temporaries" begin
    T = Float64
    Nr = 24
    cd = ChebyshevDiffn(Nr, T[0.35, 1.0], 4)
    theta_coeffs = Dict{Int, Vector{T}}(
        ℓ => fill(T(0.02) / T(ℓ + 1), Nr) for ℓ in 1:3)

    uphi_coeffs = Dict{Int, Vector{T}}()
    duphi_dr_coeffs = Dict{Int, Vector{T}}()
    Cross.solve_thermal_wind_coupled!(
        uphi_coeffs, duphi_dr_coeffs, theta_coeffs, 1, cd,
        T(0.35), one(T), T(100), one(T);
        E = T(1e-3), lmax = 4)

    GC.gc()
    uphi_coeffs = Dict{Int, Vector{T}}()
    duphi_dr_coeffs = Dict{Int, Vector{T}}()
    bytes = @allocated Cross.solve_thermal_wind_coupled!(
        uphi_coeffs, duphi_dr_coeffs, theta_coeffs, 1, cd,
        T(0.35), one(T), T(100), one(T);
        E = T(1e-3), lmax = 4)

    @test bytes < 170_000
end

@testset "Triglobal unweighted coupling avoids quadrature node allocation" begin
    Cross.compute_sh_coupling_unweighted(3, 1, 2, 0, 3, 1)
    GC.gc()
    bytes = @allocated Cross.compute_sh_coupling_unweighted(3, 1, 2, 0, 3, 1)

    @test bytes < 256
end

@testset "Triglobal shift-invert map reuses Krylov buffers" begin
    A = spdiagm(0 => ComplexF64[2, 3, 4, 5])
    B = spdiagm(0 => ComplexF64[1, 1, 1, 1])
    F = lu(A - (0.1 + 1e-6im) * B)
    shift_map = Cross._triglobal_shift_invert_map(F, B)
    x = ComplexF64[1, 2, 3, 4]
    y = similar(x)

    mul!(y, shift_map, x)
    @test y ≈ F \ (B * x)

    GC.gc()
    bytes = @allocated mul!(y, shift_map, x)
    @test bytes < 1_000
end

@testset "Full meridional coupled solve reuses mode-independent radial work" begin
    T = Float64
    Nr = 32
    lmax = 8
    m = 1
    cd = ChebyshevDiffn(Nr, T[0.35, 1.0], 2)
    theta_coeffs = Dict{Tuple{Int,Int}, Vector{T}}(
        (ℓ, m) => fill(T(0.05) / T(ℓ + 1), Nr) for ℓ in m:lmax)
    uphi_coeffs = Dict{Tuple{Int,Int}, Vector{T}}()

    function run_meridional(theta_coeffs, uphi_coeffs, cd)
        ur_coeffs = Dict{Tuple{Int,Int}, Vector{T}}()
        utheta_coeffs = Dict{Tuple{Int,Int}, Vector{T}}()
        dur_dr_coeffs = Dict{Tuple{Int,Int}, Vector{T}}()
        dutheta_dr_coeffs = Dict{Tuple{Int,Int}, Vector{T}}()
        Cross.solve_meridional_coupled!(
            ur_coeffs, utheta_coeffs, dur_dr_coeffs, dutheta_dr_coeffs,
            theta_coeffs, uphi_coeffs, cd.x, cd.D1, cd.D2,
            T(0.35), one(T), T(100), T(1e-3), one(T), m, lmax)
        return ur_coeffs, utheta_coeffs
    end

    run_meridional(theta_coeffs, uphi_coeffs, cd)
    GC.gc()
    bytes = @allocated run_meridional(theta_coeffs, uphi_coeffs, cd)

    @test bytes < 1_300_000
end

@testset "Triglobal shift-invert handles Float32 sparse LU promotion" begin
    A = spdiagm(0 => ComplexF32[2, 3, 4, 5])
    B = spdiagm(0 => ComplexF32[1, 1, 1, 1])
    F = lu(A - (0.1f0 + 1f-6im) * B)
    shift_map = Cross._triglobal_shift_invert_map(F, B)
    x = ComplexF32[1, 2, 3, 4]
    y = similar(x)

    result_or_error = try
        mul!(y, shift_map, x)
        nothing
    catch err
        err
    end

    @test result_or_error === nothing
    @test y ≈ ComplexF32.(F \ (B * x))
end

@testset "Public triglobal solve preserves Float32 result storage" begin
    T = Float32
    params = OnsetParams(
        E = T(1e-3),
        Pr = one(T),
        Ra = T(100),
        χ = T(0.35),
        m = 0,
        lmax = 3,
        Nr = 8
    )
    cd = ChebyshevDiffn(params.Nr, T[params.χ, one(T)], 1)
    empty = Dict{Tuple{Int,Int}, Vector{T}}()
    bs3d = BasicState3D{T}(
        lmax_bs = 0,
        mmax_bs = 0,
        Nr = params.Nr,
        r = cd.x,
        theta_coeffs = empty,
        dtheta_dr_coeffs = copy(empty),
        ur_coeffs = copy(empty),
        utheta_coeffs = copy(empty),
        uphi_coeffs = copy(empty),
        dur_dr_coeffs = copy(empty),
        dutheta_dr_coeffs = copy(empty),
        duphi_dr_coeffs = copy(empty)
    )

    result_or_error = try
        solve(TriglobalProblem(params, bs3d, 0:1); nev=1, sigma=zero(T), verbose=false)
    catch err
        err
    end

    @test !(result_or_error isa Exception)
    if !(result_or_error isa Exception)
        @test eltype(result_or_error.eigenvalues) === ComplexF32
        @test eltype(result_or_error.eigenvectors) === ComplexF32
    end
end

@testset "Symbolic spherical harmonic constructors preserve amplitude precision" begin
    @test typeof(Y10(1.0f0)) === SphericalHarmonicBC{Float32}
    @test typeof(Ylm(2, 1, 1.0f0)) === SphericalHarmonicBC{Float32}
end

@testset "Ultraspherical multiplication helpers preserve Float32 intermediates" begin
    c = Cross.csl([0, 1, 2], Float32(1), 3, 2)
    @test eltype(c) === Float32

    a0 = zeros(Float32, 32)
    a0[1] = 1
    a0[3] = 0.25f0
    Cross.multiplication_matrix(a0, Float32(1), 32)
    GC.gc()
    bytes = @allocated Cross.multiplication_matrix(a0, Float32(1), 32)

    @test bytes < 80_000
end

@testset "Ultraspherical multiplication streams Gegenbauer indices" begin
    a0 = zeros(Float32, 128)
    a0[1] = 1
    a0[3] = 0.25f0
    Cross.multiplication_matrix(a0, Float32(1), 128)
    GC.gc()
    bytes = @allocated Cross.multiplication_matrix(a0, Float32(1), 128)

    @test bytes < 290_000
end

@testset "Basic state operator assembly avoids dense diagonal temporaries" begin
    T = Float64
    Nr = 24
    cd = ChebyshevDiffn(Nr, T[0.35, 1.0], 4)
    r = cd.x
    theta_coeffs = Dict{Int, Vector{T}}(
        ℓ => fill(T(0.02) / T(ℓ + 1), Nr) for ℓ in 0:4)
    uphi_coeffs = Dict{Int, Vector{T}}(
        ℓ => fill(T(0.03) / T(ℓ + 1), Nr) for ℓ in 0:4)
    bs = BasicState{T}(
        lmax_bs = 4,
        Nr = Nr,
        r = r,
        theta_coeffs = theta_coeffs,
        uphi_coeffs = uphi_coeffs,
        dtheta_dr_coeffs = Dict(ℓ => cd.D1 * v for (ℓ, v) in theta_coeffs),
        duphi_dr_coeffs = Dict(ℓ => cd.D1 * v for (ℓ, v) in uphi_coeffs)
    )
    params = OnsetParams(
        E = T(1e-3),
        Pr = one(T),
        Ra = T(100),
        χ = T(0.35),
        m = 1,
        lmax = 5,
        Nr = Nr,
        basic_state = bs
    )
    op = LinearStabilityOperator(params)
    run_build() = with_logger(NullLogger()) do
        Cross.build_basic_state_operators(bs, op, params.m)
    end

    run_build()
    GC.gc()
    bytes = @allocated run_build()

    @test bytes < 4_000_000
end

@testset "Self-consistent basic state avoids avoidable vector temporaries" begin
    T = Float64
    cd = ChebyshevDiffn(24, T[0.35, 1.0], 4)
    bc = Y22(T(0.02))
    Cross.basic_state_selfconsistent(
        cd, T(0.35), T(1e-3), T(100), one(T);
        temperature_bc = bc,
        lmax_bs = 4,
        max_iterations = 1,
        verbose = false
    )

    GC.gc()
    bytes = @allocated Cross.basic_state_selfconsistent(
        cd, T(0.35), T(1e-3), T(100), one(T);
        temperature_bc = bc,
        lmax_bs = 4,
        max_iterations = 1,
        verbose = false
    )

    @test bytes < 1_200_000
end
