module Cross

    using LinearAlgebra
    using SparseArrays
    using JLD2
    using Printf

    using Parameters

    using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM

    using Arpack
    using KrylovKit

    include("Chebyshev.jl")
    include("banner.jl")
    include("get_velocity.jl")
    include("basic_state.jl")
    include("linear_stability.jl")
    include("triglobal_stability.jl")

    println(CROSS_BANNER)

    export
        ChebyshevDiffn,
        potentials_to_velocity,
        velocity_fields_from_poloidal_toroidal,
        temperature_field_from_coefficients,
        fields_from_coefficients,
        OnsetParams,
        LinearStabilityOperator,
        solve_eigenvalue_problem,
        find_growth_rate,
        find_critical_rayleigh,
        print_cross_header,
        CROSS_BANNER,
        BasicState,
        conduction_basic_state,
        meridional_basic_state,
        BasicState3D,
        nonaxisymmetric_basic_state,
        TriGlobalParams,
        setup_coupled_mode_problem,
        estimate_triglobal_problem_size,
        solve_triglobal_eigenvalue_problem,
        find_critical_rayleigh_triglobal

end
