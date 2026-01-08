module Cross

    using LinearAlgebra
    using SparseArrays
    using JLD2
    using Printf

    using Parameters

    using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM

    using KrylovKit

    include("Chebyshev.jl")
    include("banner.jl")
    include("get_velocity.jl")
    include("basic_state.jl")
    include("basic_state_operators.jl")
    include("linear_stability.jl")

    # Three analysis modes (dedicated modules)
    include("onset_convection.jl")       # Onset with no mean flow
    include("biglobal_stability.jl")     # Biglobal with axisymmetric mean flow
    include("triglobal_stability.jl")    # Triglobal with non-axisymmetric mean flow

    println(CROSS_BANNER)

    export
        # Core utilities
        ChebyshevDiffn,
        potentials_to_velocity,
        velocity_fields_from_poloidal_toroidal,
        temperature_field_from_coefficients,
        fields_from_coefficients,
        print_cross_header,
        CROSS_BANNER,

        # Base types and functions (shared)
        OnsetParams,
        ShellParams,
        LinearStabilityOperator,
        solve_eigenvalue_problem,
        leading_modes,
        find_growth_rate,
        find_critical_rayleigh,
        assemble_matrices,

        # Basic state structures
        BasicState,
        conduction_basic_state,
        meridional_basic_state,
        BasicState3D,
        nonaxisymmetric_basic_state,
        BasicStateOperators,
        build_basic_state_operators,
        add_basic_state_operators!,

        # =================================================================
        # Onset Convection (No Mean Flow)
        # =================================================================
        OnsetConvectionParams,
        solve_onset_problem,
        find_critical_Ra_onset,
        find_global_critical_onset,
        estimate_onset_problem_size,
        onset_scaling_laws,

        # =================================================================
        # Biglobal Stability (Axisymmetric Mean Flow)
        # =================================================================
        BiglobalParams,
        create_conduction_basic_state,
        create_thermal_wind_basic_state,
        create_custom_basic_state,
        solve_biglobal_problem,
        find_critical_Ra_biglobal,
        compare_onset_vs_biglobal,
        sweep_thermal_wind_amplitude,
        analyze_basic_state,

        # =================================================================
        # Triglobal Stability (Non-Axisymmetric Mean Flow)
        # =================================================================
        TriglobalParams,
        setup_coupled_mode_problem,
        estimate_triglobal_problem_size,
        solve_triglobal_eigenvalue_problem,
        find_critical_rayleigh_triglobal

end
