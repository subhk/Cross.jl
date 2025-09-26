module Cross

    using LinearAlgebra
    using SparseArrays
    using FFTW
    using JLD2
    
    using Parameters
    using BenchmarkTools

    using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM

    using Arpack
    using LinearMaps

    using KrylovKit

    using LinearAlgebra
    using Statistics

    export

        ChebyshevDiffn,

        potentials_to_velocity,

        ShellParams,
        build_generalized_problem,
        leading_modes,
        critical_Rayleigh_search


    include("Chebyshev.jl")
    include("get_velocity.jl")
    include("linear_stability.jl")

    using .LinearStability: ShellParams, build_generalized_problem, leading_modes, critical_Rayleigh_search

end
