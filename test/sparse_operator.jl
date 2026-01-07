using Test

include("../src/SparseOperator.jl")

function _expected_matrix_size(params::SparseOperator.SparseOnsetParams{T}) where {T}
    ll_top, ll_bot = SparseOperator.compute_l_modes(params.m, params.lmax, params.symm)
    n_per_mode = params.N + 1
    return (2 * length(ll_top) + length(ll_bot)) * n_per_mode
end

@testset "SparseOperator matrix sizing" begin
    cases = [
        SparseOperator.SparseOnsetParams(E=1e-4, Pr=1.0, Ra=1e6,
                                         ricb=0.35, m=1, lmax=6,
                                         symm=1, N=6),
        SparseOperator.SparseOnsetParams(E=1e-4, Pr=1.0, Ra=1e6,
                                         ricb=0.35, m=0, lmax=6,
                                         symm=1, N=6),
        SparseOperator.SparseOnsetParams(E=1e-4, Pr=1.0, Ra=1e6,
                                         ricb=0.35, m=2, lmax=7,
                                         symm=-1, N=8),
    ]

    for params in cases
        op = SparseOperator.SparseStabilityOperator(params)
        @test op.matrix_size == _expected_matrix_size(params)
    end
end
