using Test
using Cross
using LinearAlgebra

function _reduced_dense_operator(params)
    op = LinearStabilityOperator(params)
    A, B, interior_dofs, boundary_dofs = assemble_matrices(op)
    A_red, B_red, reduction = Cross._constrained_reduced_matrices(
        A, B, op, interior_dofs, boundary_dofs)
    return A_red, B_red, reduction
end

function _zero_basic_state_3d(cd, lmax_bs::Int, mmax_bs::Int)
    T = eltype(cd.x)
    coeffs = Dict{Tuple{Int,Int}, Vector{T}}()
    return BasicState3D(
        lmax_bs = lmax_bs,
        mmax_bs = mmax_bs,
        Nr = length(cd.x),
        r = cd.x,
        theta_coeffs = coeffs,
        dtheta_dr_coeffs = deepcopy(coeffs),
        ur_coeffs = deepcopy(coeffs),
        utheta_coeffs = deepcopy(coeffs),
        uphi_coeffs = deepcopy(coeffs),
        dur_dr_coeffs = deepcopy(coeffs),
        dutheta_dr_coeffs = deepcopy(coeffs),
        duphi_dr_coeffs = deepcopy(coeffs)
    )
end

@testset "Mean-flow stability operator regressions" begin
    @testset "Explicit conduction basic state matches built-in conduction" begin
        E = 1e-3
        Pr = 1.0
        Ra = 1e5
        χ = 0.35
        m = 2
        lmax = 6
        Nr = 16

        cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
        bs = conduction_basic_state(cd, χ, 6)

        params_builtin = OnsetParams(
            E = E, Pr = Pr, Ra = Ra, χ = χ,
            m = m, lmax = lmax, Nr = Nr,
            basic_state = nothing
        )
        params_explicit = OnsetParams(
            E = E, Pr = Pr, Ra = Ra, χ = χ,
            m = m, lmax = lmax, Nr = Nr,
            basic_state = bs
        )

        A_builtin, B_builtin, _ = _reduced_dense_operator(params_builtin)
        A_explicit, B_explicit, _ = _reduced_dense_operator(params_explicit)

        @test norm(B_builtin - B_explicit) <= 1e-12 * max(norm(B_builtin), 1.0)
        @test norm(A_builtin - A_explicit) <= 1e-10 * max(norm(A_builtin), 1.0)
    end

    @testset "Triglobal single-mode blocks use constraint-preserving reduction" begin
        E = 1e-3
        Pr = 1.0
        Ra = 1e4
        χ = 0.35
        m = 1
        lmax = 6
        Nr = 12

        cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
        bs3d = _zero_basic_state_3d(cd, 0, 0)
        params_tri = TriglobalParams(
            E = E, Pr = Pr, Ra = Ra, χ = χ,
            m_range = m:m,
            lmax = lmax,
            Nr = Nr,
            basic_state_3d = bs3d,
            mechanical_bc = :no_slip,
            thermal_bc = :fixed_temperature
        )

        problem = setup_coupled_mode_problem(params_tri)
        single_mode_ops = Cross.build_single_mode_operators(problem, false)

        params_single = OnsetParams(
            E = E, Pr = Pr, Ra = Ra, χ = χ,
            m = m, lmax = lmax, Nr = Nr,
            mechanical_bc = :no_slip,
            thermal_bc = :fixed_temperature,
            basic_state = nothing
        )
        A_expected, B_expected, _ = _reduced_dense_operator(params_single)

        @test norm(single_mode_ops[m].B - B_expected) <= 1e-12 * max(norm(B_expected), 1.0)
        @test norm(single_mode_ops[m].A - A_expected) <= 1e-10 * max(norm(A_expected), 1.0)
    end

    @testset "Triglobal coefficient extraction reconstructs constrained modes" begin
        E = 1e-3
        Pr = 1.0
        Ra = 1e4
        χ = 0.35
        m = 1
        lmax = 5
        Nr = 12

        cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
        bs3d = _zero_basic_state_3d(cd, 0, 0)
        params_tri = TriglobalParams(
            E = E, Pr = Pr, Ra = Ra, χ = χ,
            m_range = m:m,
            lmax = lmax,
            Nr = Nr,
            basic_state_3d = bs3d,
            mechanical_bc = :no_slip,
            thermal_bc = :fixed_temperature
        )

        problem = setup_coupled_mode_problem(params_tri)
        single_mode_ops = Cross.build_single_mode_operators(problem, false)
        block_range = problem.block_indices[m]
        block_vec = ComplexF64[complex(sin(i), cos(i)) for i in 1:length(block_range)]
        eigenvector = zeros(ComplexF64, problem.total_dofs)
        eigenvector[block_range] .= block_vec

        P_coeffs, T_coeffs = Cross._extract_mode_coefficients(eigenvector, problem, m)
        full_vec = Cross._reconstruct_full_vector(single_mode_ops[m].reduction, block_vec)
        op = single_mode_ops[m].op

        for ℓ in op.l_sets[:P]
            @test P_coeffs[ℓ] ≈ full_vec[op.index_map[(ℓ, :P)]]
        end
        for ℓ in op.l_sets[:T]
            @test T_coeffs[ℓ] ≈ full_vec[op.index_map[(ℓ, :T)]]
        end
    end
end
