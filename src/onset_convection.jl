# =============================================================================
#  Onset of Convection with No Mean Flow
#
#  Classical linear stability analysis for thermal convection in rotating
#  spherical shells with a conductive temperature profile and zero background
#  flow.
#
#  This is the simplest stability analysis mode:
#  - Base state: θ̄(r) = conductive profile, ū = 0
#  - Each azimuthal mode m is independent (no mode coupling)
#  - Find critical Rayleigh number Ra_c(m) and minimum over all m
#
#  Physical problem:
#  ----------------
#  Linearized Boussinesq equations about the conductive state:
#
#    ∂u'/∂t + 2Ω̂×u' = -∇p' + E∇²u' + (Ra E²/Pr) Θ' r̂
#    ∂Θ'/∂t + u'·∇θ̄ = (E/Pr) ∇²Θ'
#    ∇·u' = 0
#
#  Eigenvalue problem:
#    A x = σ B x
#
#  where σ = σ_r + iω is the complex growth rate (σ_r > 0 → unstable).
# =============================================================================

using Parameters
using LinearAlgebra
using Printf

# Import from parent module
import ..Cross: LinearStabilityOperator, OnsetParams, assemble_matrices,
                solve_eigenvalue_problem, find_growth_rate, ChebyshevDiffn

"""
    OnsetConvectionParams{T<:Real}

Parameters for classical onset of convection analysis (no mean flow).

This is a simplified interface for the most common use case: finding the
critical Rayleigh number for the onset of thermal convection in a rotating
spherical shell with conductive base state.

# Fields
- `E::T` - Ekman number (viscous/Coriolis ratio)
- `Pr::T` - Prandtl number (momentum/thermal diffusivity ratio)
- `Ra::T` - Rayleigh number (buoyancy forcing strength)
- `χ::T` - Radius ratio r_i/r_o
- `m::Int` - Azimuthal wavenumber of perturbation
- `lmax::Int` - Maximum spherical harmonic degree
- `Nr::Int` - Number of radial collocation points
- `mechanical_bc::Symbol` - :no_slip or :stress_free
- `thermal_bc::Symbol` - :fixed_temperature or :fixed_flux
- `equatorial_symmetry::Symbol` - :both, :symmetric, or :antisymmetric

# Example
```julia
params = OnsetConvectionParams(
    E = 1e-5,
    Pr = 1.0,
    Ra = 1e7,
    χ = 0.35,
    m = 10,
    lmax = 60,
    Nr = 64
)
```

See also: [`solve_onset_problem`](@ref), [`find_critical_Ra_onset`](@ref)
"""
@with_kw struct OnsetConvectionParams{T<:Real}
    E::T
    Pr::T = one(T)
    Ra::T
    χ::T
    m::Int
    lmax::Int
    Nr::Int
    mechanical_bc::Symbol = :no_slip
    thermal_bc::Symbol = :fixed_temperature
    equatorial_symmetry::Symbol = :both

    function OnsetConvectionParams{T}(E, Pr, Ra, χ, m, lmax, Nr,
                                       mechanical_bc, thermal_bc,
                                       equatorial_symmetry) where T
        @assert 0 < χ < 1 "Radius ratio must satisfy 0 < χ < 1"
        @assert E > 0 "Ekman number must be positive"
        @assert Pr > 0 "Prandtl number must be positive"
        @assert m >= 0 "Azimuthal wavenumber must be non-negative"
        @assert lmax >= m "lmax must be >= m"
        @assert Nr >= 4 "Need at least 4 radial points"
        @assert mechanical_bc in (:no_slip, :stress_free) "Invalid mechanical BC"
        @assert thermal_bc in (:fixed_temperature, :fixed_flux) "Invalid thermal BC"
        @assert equatorial_symmetry in (:both, :symmetric, :antisymmetric)

        new{T}(E, Pr, Ra, χ, m, lmax, Nr, mechanical_bc, thermal_bc, equatorial_symmetry)
    end
end


"""
    solve_onset_problem(params::OnsetConvectionParams; nev=6, kwargs...)

Solve the onset of convection eigenvalue problem.

Computes eigenvalues σ = σ_r + iω where:
- σ_r > 0: unstable (growing perturbation)
- σ_r = 0: marginal stability (onset)
- σ_r < 0: stable (decaying perturbation)
- ω: drift frequency (pattern rotation rate)

# Arguments
- `params::OnsetConvectionParams` - Problem parameters
- `nev::Int` - Number of eigenvalues to compute (default: 6)
- `tol::Float64` - Eigenvalue solver tolerance (default: 1e-10)
- `which::Symbol` - Target eigenvalues: :LR (largest real), :LM (largest magnitude)

# Returns
- `eigenvalues::Vector{ComplexF64}` - Complex growth rates (sorted by real part)
- `eigenvectors::Vector{Vector{ComplexF64}}` - Corresponding eigenmodes
- `operator::LinearStabilityOperator` - The assembled operator
- `info` - Solver convergence information

# Example
```julia
params = OnsetConvectionParams(E=1e-5, Pr=1.0, Ra=1e7, χ=0.35, m=10, lmax=60, Nr=64)
eigenvalues, eigenvectors, op, info = solve_onset_problem(params; nev=8)

σ₁ = real(eigenvalues[1])
ω₁ = imag(eigenvalues[1])
println("Leading mode: σ = \$σ₁, ω = \$ω₁")
```
"""
function solve_onset_problem(params::OnsetConvectionParams{T};
                             nev::Int=6,
                             tol::Float64=1e-10,
                             maxiter::Int=1000,
                             which::Symbol=:LR,
                             sigma=nothing) where T

    # Convert to internal OnsetParams (no basic_state → pure conduction)
    internal_params = OnsetParams(
        E = params.E,
        Pr = params.Pr,
        Ra = params.Ra,
        χ = params.χ,
        m = params.m,
        lmax = params.lmax,
        Nr = params.Nr,
        mechanical_bc = params.mechanical_bc,
        thermal_bc = params.thermal_bc,
        equatorial_symmetry = params.equatorial_symmetry,
        basic_state = nothing  # No basic state = conduction profile
    )

    # Build operator and solve
    op = LinearStabilityOperator(internal_params)
    eigenvalues, eigenvectors, info = solve_eigenvalue_problem(op;
        nev=nev, tol=tol, maxiter=maxiter, which=which, sigma=sigma)

    return eigenvalues, eigenvectors, op, info
end


"""
    find_critical_Ra_onset(; E, Pr, χ, m, lmax, Nr, kwargs...)

Find the critical Rayleigh number for onset of convection at a specific m.

Uses bisection to find Ra_c where the leading growth rate σ = 0.

# Arguments
- `E::Real` - Ekman number
- `Pr::Real` - Prandtl number
- `χ::Real` - Radius ratio
- `m::Int` - Azimuthal wavenumber
- `lmax::Int` - Maximum spherical harmonic degree
- `Nr::Int` - Number of radial points
- `Ra_guess::Real` - Initial guess for Ra_c (default: 1e6)
- `tol::Real` - Tolerance for convergence (default: 1e-6)
- `mechanical_bc::Symbol` - Boundary conditions (default: :no_slip)
- `thermal_bc::Symbol` - Thermal boundary conditions (default: :fixed_temperature)

# Returns
- `Ra_c::Real` - Critical Rayleigh number
- `ω_c::Real` - Drift frequency at onset
- `eigenvector` - Critical eigenmode

# Example
```julia
Ra_c, ω_c, vec = find_critical_Ra_onset(
    E = 1e-5, Pr = 1.0, χ = 0.35, m = 10,
    lmax = 60, Nr = 64, Ra_guess = 1e7
)
println("Critical Ra for m=10: Ra_c = \$Ra_c, ω_c = \$ω_c")
```

See also: [`find_global_critical_onset`](@ref)
"""
function find_critical_Ra_onset(; E::T, Pr::T, χ::T, m::Int, lmax::Int, Nr::Int,
                                 Ra_guess::T=T(1e6),
                                 tol::T=T(1e-6),
                                 Ra_bracket::Tuple{T,T}=(Ra_guess/10, Ra_guess*10),
                                 mechanical_bc::Symbol=:no_slip,
                                 thermal_bc::Symbol=:fixed_temperature,
                                 equatorial_symmetry::Symbol=:both,
                                 nev::Int=6,
                                 verbose::Bool=false) where {T<:Real}

    # Use the existing find_critical_rayleigh function from linear_stability.jl
    # but with no basic_state
    Ra_c, ω_c, vec_c = Cross.find_critical_rayleigh(
        E, Pr, χ, m, lmax, Nr;
        Ra_guess=Ra_guess, tol=tol, Ra_bracket=Ra_bracket,
        mechanical_bc=mechanical_bc, thermal_bc=thermal_bc,
        equatorial_symmetry=equatorial_symmetry, nev=nev
    )

    if verbose
        @printf("  m = %d: Ra_c = %.6e, ω_c = %+.6f\n", m, Ra_c, ω_c)
    end

    return Ra_c, ω_c, vec_c
end


"""
    find_global_critical_onset(; E, Pr, χ, lmax, Nr, m_range, kwargs...)

Find the global critical Rayleigh number by sweeping over azimuthal modes.

The global critical Rayleigh number is the minimum Ra_c across all m:
    Ra_c^global = min_m Ra_c(m)

# Arguments
- `E::Real` - Ekman number
- `Pr::Real` - Prandtl number
- `χ::Real` - Radius ratio
- `lmax::Int` - Maximum spherical harmonic degree
- `Nr::Int` - Number of radial points
- `m_range` - Range of azimuthal modes to scan (e.g., 5:25)
- `Ra_guess::Real` - Initial guess for Ra_c (default: 1e6)
- `tol::Real` - Tolerance for each m (default: 1e-6)
- `verbose::Bool` - Print progress (default: true)
- `equatorial_symmetry::Symbol` - :both, :symmetric, or :antisymmetric

# Returns
- `m_c::Int` - Critical azimuthal wavenumber
- `Ra_c::Real` - Global critical Rayleigh number
- `ω_c::Real` - Drift frequency at onset
- `all_results::Dict` - Results for all m values

# Example
```julia
m_c, Ra_c, ω_c, results = find_global_critical_onset(
    E = 1e-5, Pr = 1.0, χ = 0.35,
    lmax = 60, Nr = 64, m_range = 5:25
)
println("Global critical: m_c = \$m_c, Ra_c = \$Ra_c")
```

# Scaling Laws
At low Ekman number, theory predicts:
- Ra_c ~ C_Ra × E^(-4/3)
- m_c ~ C_m × E^(-1/3)
- ω_c ~ C_ω × E^(-2/3)
"""
function find_global_critical_onset(; E::T, Pr::T, χ::T, lmax::Int, Nr::Int,
                                     m_range::AbstractRange,
                                     Ra_guess::T=T(1e6),
                                     tol::T=T(1e-6),
                                     mechanical_bc::Symbol=:no_slip,
                                     thermal_bc::Symbol=:fixed_temperature,
                                     equatorial_symmetry::Symbol=:both,
                                     verbose::Bool=true) where {T<:Real}
    @assert all(m -> m >= 0, m_range) "m_range must be non-negative for onset analysis"
    @assert equatorial_symmetry in (:both, :symmetric, :antisymmetric) "equatorial_symmetry must be :both, :symmetric, or :antisymmetric"

    if verbose
        println("="^60)
        println("Finding Global Critical Rayleigh Number (Onset)")
        println("="^60)
        println("  E       = ", @sprintf("%.2e", E))
        println("  Pr      = ", Pr)
        println("  χ       = ", χ)
        println("  m_range = ", m_range)
        println()
    end

    results = Dict{Int, NamedTuple{(:Ra_c, :ω_c), Tuple{T, T}}}()

    if verbose
        @printf("  %-4s  %-14s  %-14s\n", "m", "Ra_c", "ω_c")
        println("  " * "-"^35)
    end

    for m in m_range
        try
            Ra_c, ω_c, _ = find_critical_Ra_onset(
                E=E, Pr=Pr, χ=χ, m=m,
                lmax=max(lmax, m + 10),
                Nr=Nr,
                Ra_guess=Ra_guess,
                tol=tol,
                mechanical_bc=mechanical_bc,
                thermal_bc=thermal_bc,
                equatorial_symmetry=equatorial_symmetry
            )
            results[m] = (Ra_c=Ra_c, ω_c=ω_c)

            if verbose
                @printf("  %-4d  %.8e  %+.8f\n", m, Ra_c, ω_c)
            end

            # Update guess for next m (Ra_c changes smoothly with m)
            Ra_guess = Ra_c

        catch err
            if verbose
                @printf("  %-4d  FAILED\n", m)
            end
            results[m] = (Ra_c=T(NaN), ω_c=T(NaN))
        end
    end

    # Find global minimum
    valid_results = filter(p -> !isnan(p.second.Ra_c), results)

    if isempty(valid_results)
        error("No valid results found in the m_range")
    end

    _, m_c = findmin(m -> results[m].Ra_c, keys(valid_results))
    Ra_c = results[m_c].Ra_c
    ω_c = results[m_c].ω_c

    if verbose
        println()
        println("="^60)
        println("Global Critical Parameters")
        println("="^60)
        @printf("  Critical mode:     m_c  = %d\n", m_c)
        @printf("  Critical Rayleigh: Ra_c = %.8e\n", Ra_c)
        @printf("  Drift frequency:   ω_c  = %+.8f\n", ω_c)
        println()

        # Scaling coefficients
        Ra_coeff = Ra_c * E^(4/3)
        m_coeff = m_c * E^(1/3)
        @printf("  Scaling coefficients:\n")
        @printf("    Ra_c × E^(4/3) = %.4f\n", Ra_coeff)
        @printf("    m_c × E^(1/3)  = %.4f\n", m_coeff)
    end

    return m_c, Ra_c, ω_c, results
end


"""
    estimate_onset_problem_size(params::OnsetConvectionParams)

Estimate the size of the onset eigenvalue problem.

# Returns
- `total_dofs::Int` - Total degrees of freedom
- `matrix_size::Int` - Size of the matrices
- `num_ell_modes::Int` - Number of spherical harmonic modes
- `memory_estimate_mb::Float64` - Estimated memory in MB

# Example
```julia
params = OnsetConvectionParams(E=1e-5, Pr=1.0, Ra=1e7, χ=0.35, m=10, lmax=60, Nr=64)
size_info = estimate_onset_problem_size(params)
println("Problem size: \$(size_info.total_dofs) DOFs, ~\$(size_info.memory_estimate_mb) MB")
```
"""
function estimate_onset_problem_size(params::OnsetConvectionParams)
    m = params.m
    lmax = params.lmax
    Nr = params.Nr

    # Account for equatorial symmetry when counting modes.
    internal_params = OnsetParams(
        E = params.E,
        Pr = params.Pr,
        Ra = params.Ra,
        χ = params.χ,
        m = params.m,
        lmax = params.lmax,
        Nr = params.Nr,
        mechanical_bc = params.mechanical_bc,
        thermal_bc = params.thermal_bc,
        equatorial_symmetry = params.equatorial_symmetry,
        basic_state = nothing
    )
    l_sets = compute_l_sets(internal_params)
    total_dofs = (length(l_sets[:P]) + length(l_sets[:T]) + length(l_sets[:Θ])) * Nr
    matrix_size = total_dofs

    # Memory: A and B matrices (complex, dense for now)
    # Each matrix: N × N × 16 bytes (ComplexF64)
    memory_bytes = 2 * matrix_size^2 * 16
    memory_mb = memory_bytes / (1024^2)

    return (
        total_dofs = total_dofs,
        matrix_size = matrix_size,
        num_ell_modes = length(l_sets[:Θ]),
        memory_estimate_mb = memory_mb
    )
end


"""
    onset_scaling_laws(E::Real, χ::Real; bc::Symbol=:no_slip)

Estimate critical parameters from asymptotic scaling laws.

For low Ekman number E << 1, theory predicts power-law scalings:

| Quantity | Scaling |
|----------|---------|
| Ra_c     | C_Ra × E^(-4/3) |
| m_c      | C_m × E^(-1/3) |
| ω_c      | C_ω × E^(-2/3) |
| δ        | C_δ × E^(1/3) |

where δ is the convection column width.

# Arguments
- `E::Real` - Ekman number
- `χ::Real` - Radius ratio
- `bc::Symbol` - Boundary conditions (:no_slip or :stress_free)

# Returns
Named tuple with estimated Ra_c, m_c, ω_c, δ

Note: These are rough estimates based on asymptotic theory.
Numerical computation is needed for accurate values.
"""
function onset_scaling_laws(E::T, χ::T; bc::Symbol=:no_slip) where {T<:Real}
    # Coefficients depend on χ and BC type
    # These are approximate values from literature

    if bc == :no_slip
        C_Ra = T(6.0)   # Approximate for χ ≈ 0.35
        C_m = T(0.5)
        C_ω = T(0.4)
    else  # stress_free
        C_Ra = T(4.0)
        C_m = T(0.5)
        C_ω = T(0.5)
    end

    Ra_c_est = C_Ra * E^(-4/3)
    m_c_est = round(Int, C_m * E^(-1/3))
    ω_c_est = C_ω * E^(-2/3)
    δ_est = E^(1/3)

    return (
        Ra_c = Ra_c_est,
        m_c = m_c_est,
        ω_c = ω_c_est,
        δ = δ_est
    )
end


# =============================================================================
#  Exports
# =============================================================================

export OnsetConvectionParams
export solve_onset_problem
export find_critical_Ra_onset
export find_global_critical_onset
export estimate_onset_problem_size
export onset_scaling_laws
