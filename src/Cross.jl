module Cross

    using LinearAlgebra
    using SparseArrays
    using JLD2

    using Parameters

    using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM

    using Arpack
    using KrylovKit

    export

        ChebyshevDiffn,

        potentials_to_velocity,
        velocity_fields_from_poloidal_toroidal,
        temperature_field_from_coefficients,
        fields_from_coefficients,

        ShellParams,
        MeridionalOperator,
        setup_operator,
        leading_modes,
        critical_rayleigh,
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
