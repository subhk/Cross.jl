module Cross

    using LinearAlgebra
    using SparseArrays
    using JLD2

    using Parameters
    using BenchmarkTools

    using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM

    using Arpack
    using KrylovKit

    # Conditional loading of Statistics for Julia 1.10.10 compatibility
    if VERSION >= v"1.11"
        using Statistics
    else
        # For Julia 1.10.10, try to load Statistics carefully
        try
            using Statistics
        catch e
            @warn "Could not load Statistics package: $e. Some functionality may be limited."
        end
    end

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