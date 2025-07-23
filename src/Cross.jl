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

        potentials_to_velocity


    include("Chebyshev.jl")
    include("get_velocity.jl")

end