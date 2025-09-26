module Cross

    using LinearAlgebra
    using SparseArrays
    using JLD2

    using Parameters
    using BenchmarkTools

    using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM

    using Arpack
    using KrylovKit

    using Statistics

    export

        ChebyshevDiffn,

        potentials_to_velocity,

        ShellParams,
        MeridionalOperator,
        setup_operator,
        leading_modes,
        apply_operator,
        apply_mass


    include("Chebyshev.jl")
    include("get_velocity.jl")
    include("linear_stability.jl")

    using .LinearStability: ShellParams,
                            MeridionalOperator,
                            setup_operator,
                            leading_modes,
                            apply_operator,
                            apply_mass

end
