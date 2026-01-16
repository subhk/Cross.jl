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
    include("advection_diffusion.jl")  # Self-consistent basic state solver
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
        BasicState3D,
        conduction_basic_state,
        meridional_basic_state,
        nonaxisymmetric_basic_state,
        basic_state,  # High-level convenience function

        # Self-consistent basic state (with advection)
        nonaxisymmetric_basic_state_selfconsistent,
        basic_state_selfconsistent,
        AdvectionDiffusionSolver,
        compute_phi_advection_spectral,
        compute_full_advection_spectral,
        solve_poisson_mode,

        # Meridional circulation (toroidal-poloidal decomposition)
        solve_meridional_coupled!,    # Full block-tridiagonal solver (exact)
        solve_meridional_simple!,     # Diagonal approximation (fast)
        solve_meridional_circulation_toroidal_poloidal!,  # Main wrapper
        sin_theta_coupling,
        cos_theta_coupling,
        theta_derivative_coupling,    # sinθ × ∂Y/∂θ coupling coefficients
        inv_sin_theta_gaunt,          # ⟨Y_Lm|1/sinθ|Y_ℓm⟩ integrals
        inv_sin_theta_coupling,       # Approximate 1/sinθ coupling

        # Symbolic spherical harmonic boundary conditions
        SphericalHarmonicBC,
        Ylm,  # General constructor
        Y00, Y10, Y11,  # Monopole and dipole
        Y20, Y21, Y22,  # Quadrupole
        Y30, Y31, Y32, Y33,  # Octupole
        Y40, Y41, Y42, Y43, Y44,  # Hexadecapole
        to_dict,
        get_lmax, get_mmax, get_lmax_mmax,
        is_axisymmetric,

        # Thermal wind solvers
        solve_thermal_wind_balance!,
        solve_thermal_wind_balance_3d!,
        build_thermal_wind,
        build_thermal_wind_3d,
        theta_derivative_coeff_3d,

        # Basic state operators
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
