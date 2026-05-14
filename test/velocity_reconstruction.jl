using Test
using Cross

function _empty_velocity_basic_state_3d(::Type{T}, Nr::Int, χ; lmax_bs::Int=0, mmax_bs::Int=0) where {T<:Real}
    cd = ChebyshevDiffn(Nr, T[T(χ), one(T)], 1)
    empty = Dict{Tuple{Int,Int}, Vector{T}}()
    return BasicState3D{T}(
        lmax_bs = lmax_bs,
        mmax_bs = mmax_bs,
        Nr = Nr,
        r = cd.x,
        theta_coeffs = empty,
        dtheta_dr_coeffs = deepcopy(empty),
        ur_coeffs = deepcopy(empty),
        utheta_coeffs = deepcopy(empty),
        uphi_coeffs = deepcopy(empty),
        dur_dr_coeffs = deepcopy(empty),
        dutheta_dr_coeffs = deepcopy(empty),
        duphi_dr_coeffs = deepcopy(empty)
    )
end

@testset "Velocity reconstruction regressions" begin
    @testset "Default meridional grid builds Gauss-Legendre nodes" begin
        grid = Cross.build_meridional_grid(12, 2, 6)

        @test length(grid.θ) == 12
        @test all(isfinite, grid.θ)
        @test all(isfinite, grid.cosθ)
        @test all(isfinite, grid.sinθ)
    end

    @testset "Hydrodynamic eigenvector reconstructs on default and supplied grids" begin
        params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
        op = LinearStabilityOperator(params)
        eigenvector = randn(ComplexF64, op.total_dof)

        ur, uθ, uφ, grid = Cross.eigenvector_to_velocity(eigenvector, op)
        @test size(ur) == (params.Nr, 2 * params.lmax)
        @test size(uθ) == size(ur)
        @test size(uφ) == size(ur)
        @test grid isa Cross.MeridionalGrid
        @test all(isfinite, ur)
        @test all(isfinite, uθ)
        @test all(isfinite, uφ)

        supplied_grid = Cross.build_meridional_grid(10, params.m, params.lmax; grid_type=:uniform)
        ur2, uθ2, uφ2, returned_grid = Cross.eigenvector_to_velocity(eigenvector, op; grid=supplied_grid)
        @test returned_grid === supplied_grid
        @test size(ur2) == (params.Nr, 10)
        @test size(uθ2) == size(ur2)
        @test size(uφ2) == size(ur2)
    end

    @testset "Triglobal reconstruction cache stores concrete field types" begin
        T = Float64
        Nr = 8
        χ = T(0.35)
        bs3d = _empty_velocity_basic_state_3d(T, Nr, χ)
        params = Cross.TriglobalParams(
            E = T(1e-3), Pr = one(T), Ra = T(100.0), χ = χ,
            m_range = -1:1,
            lmax = 2,
            Nr = Nr,
            basic_state_3d = bs3d
        )
        problem = Cross.setup_coupled_mode_problem(params)

        reconstruction = Cross._mode_reconstruction(problem, 1)

        @test isconcretetype(fieldtype(typeof(reconstruction), :op))
        @test isconcretetype(fieldtype(typeof(reconstruction), :reduction))
    end
end
