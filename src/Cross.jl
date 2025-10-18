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

        # Linear stability analysis
        OnsetParams,
        LinearStabilityOperator,
        solve_eigenvalue_problem,
        find_growth_rate,
        find_critical_rayleigh,

        # Basic state (axisymmetric)
        BasicState,
        conduction_basic_state,
        meridional_basic_state,

        # Basic state (3D / tri-global)
        BasicState3D,
        nonaxisymmetric_basic_state,

        # Tri-global stability analysis
        TriGlobalParams,
        setup_coupled_mode_problem,
        estimate_triglobal_problem_size,
        solve_triglobal_eigenvalue_problem,
        find_critical_rayleigh_triglobal


    include("Chebyshev.jl")
    include("get_velocity.jl")
    include("basic_state.jl")
    include("linear_stability.jl")
    include("triglobal_stability.jl")

end
