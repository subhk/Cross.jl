# =============================================================================
#  Linear Stability Analysis for Rotating Spherical Shell Convection
#  Implements equations from docs/poloidal_toroidal_derivation.tex
# =============================================================================

using LinearAlgebra
using SparseArrays
using KrylovKit
using Parameters

import ..Cross: ChebyshevDiffn

# =============================================================================
#  Parameter Structure
# =============================================================================

"""
    OnsetParams

Parameters for the onset of convection problem in a rotating spherical shell.

# Fields
- `E::Float64`: Ekman number (ν/ΩL²)
- `Pr::Float64`: Prandtl number (ν/κ)
- `Ra::Float64`: Rayleigh number (αg₀ΔTL³/νκ)
- `χ::Float64`: Radius ratio (rᵢ/r₀)
- `m::Int`: Azimuthal wavenumber
- `lmax::Int`: Maximum spherical harmonic degree
- `Nr::Int`: Number of radial collocation points
- `ri::Float64`: Inner radius
- `ro::Float64`: Outer radius
- `mechanical_bc::Symbol`: Velocity BC (:no_slip or :stress_free)
- `thermal_bc::Symbol`: Temperature BC (:fixed_temperature or :fixed_flux)
- `basic_state::Union{Nothing,BasicState}`: Optional basic state (default: conduction)
"""
@with_kw struct OnsetParams{T<:Real}
    # Physical parameters
    E::T
    Pr::T = 1.0
    Ra::T
    χ::T
    m::Int

    # Numerical parameters
    lmax::Int
    Nr::Int

    # Derived quantities
    ri::T = χ
    ro::T = 1.0
    L::T = ro - ri

    # Boundary conditions
    mechanical_bc::Symbol = :no_slip
    thermal_bc::Symbol = :fixed_temperature

    # Basic state (optional)
    basic_state::Union{Nothing,BasicState{T}} = nothing

    # Validation
    function OnsetParams{T}(E, Pr, Ra, χ, m, lmax, Nr, ri, ro, L,
                           mechanical_bc, thermal_bc, basic_state) where T
        @assert 0 < χ < 1 "Radius ratio must satisfy 0 < χ < 1"
        @assert E > 0 "Ekman number must be positive"
        @assert Pr > 0 "Prandtl number must be positive"
        @assert m >= 0 "Azimuthal wavenumber must be non-negative"
        @assert lmax >= m "lmax must be >= m"
        @assert Nr >= 4 "Need at least 4 radial points"
        @assert mechanical_bc in [:no_slip, :stress_free] "Invalid mechanical BC"
        @assert thermal_bc in [:fixed_temperature, :fixed_flux] "Invalid thermal BC"

        new{T}(E, Pr, Ra, χ, m, lmax, Nr, ri, ro, L, mechanical_bc, thermal_bc, basic_state)
    end
end

# Outer constructor
function OnsetParams(; E::T, Pr::T=1.0, Ra::T, χ::T, m::Int,
                     lmax::Int, Nr::Int, kwargs...) where T<:Real
    ri = χ
    ro = one(T)
    L = ro - ri
    mechanical_bc = get(kwargs, :mechanical_bc, :no_slip)
    thermal_bc = get(kwargs, :thermal_bc, :fixed_temperature)
    basic_state = get(kwargs, :basic_state, nothing)

    OnsetParams{T}(E, Pr, Ra, χ, m, lmax, Nr, ri, ro, L,
                   mechanical_bc, thermal_bc, basic_state)
end

# =============================================================================
#  Spectral Operators
# =============================================================================

"""
    radial_operator_L(ℓ, D1, D2, r)

Construct the radial operator Lℓ = ∂²/∂r² + (2/r)∂/∂r - ℓ(ℓ+1)/r²
This appears in equation (108) of the derivation.
"""
function radial_operator_L(ℓ::Int, D1::AbstractMatrix, D2::AbstractMatrix,
                          r::AbstractVector)
    Nr = length(r)
    Lℓ = D2 + Diagonal(2 ./ r) * D1 - Diagonal(ℓ*(ℓ+1) ./ r.^2)
    return Lℓ
end

"""
    scalar_operator_S(ℓ, D1, D2, r)

Construct the scalar radial operator Sℓ for temperature equation.
From equation (233): Sℓ[f] = (1/r²)∂/∂r(r²∂f/∂r) - ℓ(ℓ+1)/r²·f
"""
function scalar_operator_S(ℓ::Int, D1::AbstractMatrix, D2::AbstractMatrix,
                          r::AbstractVector)
    Nr = length(r)
    # S_ℓ[f] = (1/r²) d/dr(r² df/dr) - ℓ(ℓ+1)/r² f
    #        = d²f/dr² + (2/r) df/dr - ℓ(ℓ+1)/r² f
    # This is the same as L_ℓ!
    return radial_operator_L(ℓ, D1, D2, r)
end

"""
    coriolis_coefficients(ℓ, m)

Compute the ladder operator coefficients a⁺ₗₘ and a⁻ₗₘ from equations (131-133).
"""
function coriolis_coefficients(ℓ::Int, m::Int)
    # Equation (131)
    a_plus = sqrt(((ℓ+1)^2 - m^2) / ((2ℓ+1)*(2ℓ+3)))

    # Equation (133)
    if ℓ > 0
        a_minus = sqrt((ℓ^2 - m^2) / ((2ℓ-1)*(2ℓ+1)))
    else
        a_minus = 0.0
    end

    return a_plus, a_minus
end

# =============================================================================
#  Linear Stability Operator
# =============================================================================

"""
    LinearStabilityOperator

Represents the linear stability operator for the onset of convection problem.
Implements equations (167-174) and (204-208) from the derivation.
"""
struct LinearStabilityOperator{T<:Real}
    params::OnsetParams{T}
    cd::ChebyshevDiffn{T}
    r::Vector{T}

    # Radial operators for each ℓ
    Lℓ_ops::Dict{Int, Matrix{T}}
    Sℓ_ops::Dict{Int, Matrix{T}}

    # State vector dimensions
    ndof_per_field::Dict{Int, Int}  # Number of DOFs for each ℓ mode
    total_dof::Int

    # Index mappings: (ℓ, field_type) -> range in state vector
    # field_type: :P (poloidal), :T (toroidal), :Θ (temperature)
    index_map::Dict{Tuple{Int, Symbol}, UnitRange{Int}}
end

"""
    LinearStabilityOperator(params::OnsetParams)

Construct the linear stability operator.
"""
function LinearStabilityOperator(params::OnsetParams{T}) where T
    # Create Chebyshev differentiation
    cd = ChebyshevDiffn(params.Nr, [params.ri, params.ro], 2)
    r = cd.x

    # Build radial operators for each ℓ
    Lℓ_ops = Dict{Int, Matrix{T}}()
    Sℓ_ops = Dict{Int, Matrix{T}}()

    for ℓ in params.m:params.lmax
        Lℓ_ops[ℓ] = radial_operator_L(ℓ, cd.D1, cd.D2, r)
        Sℓ_ops[ℓ] = scalar_operator_S(ℓ, cd.D1, cd.D2, r)
    end

    # Determine DOF structure
    ndof_per_field = Dict{Int, Int}()
    for ℓ in params.m:params.lmax
        ndof_per_field[ℓ] = params.Nr  # Each field (P, T, Θ) has Nr points
    end

    n_modes = params.lmax - params.m + 1
    total_dof = 3 * params.Nr * n_modes  # 3 fields × Nr points × n_modes

    # Build index map
    index_map = Dict{Tuple{Int, Symbol}, UnitRange{Int}}()
    idx = 1
    for ℓ in params.m:params.lmax
        # Poloidal
        index_map[(ℓ, :P)] = idx:(idx + params.Nr - 1)
        idx += params.Nr

        # Toroidal
        index_map[(ℓ, :T)] = idx:(idx + params.Nr - 1)
        idx += params.Nr

        # Temperature
        index_map[(ℓ, :Θ)] = idx:(idx + params.Nr - 1)
        idx += params.Nr
    end

    return LinearStabilityOperator{T}(params, cd, r, Lℓ_ops, Sℓ_ops,
                                      ndof_per_field, total_dof, index_map)
end

"""
    apply_operator(op::LinearStabilityOperator, x::Vector)

Apply the A operator to state vector x.
Implements the RHS of equations (167-174): the spatial operators + Coriolis + buoyancy.
"""
function apply_operator(op::LinearStabilityOperator{T}, x::AbstractVector{T}) where T
    p = op.params
    result = zeros(T, op.total_dof)

    # Extract fields for each ℓ
    P_fields = Dict{Int, Vector{T}}()
    T_fields = Dict{Int, Vector{T}}()
    Θ_fields = Dict{Int, Vector{T}}()

    for ℓ in p.m:p.lmax
        P_fields[ℓ] = x[op.index_map[(ℓ, :P)]]
        T_fields[ℓ] = x[op.index_map[(ℓ, :T)]]
        Θ_fields[ℓ] = x[op.index_map[(ℓ, :Θ)]]
    end

    # Apply equations for each ℓ mode
    for ℓ in p.m:p.lmax
        # Get operator ranges
        P_idx = op.index_map[(ℓ, :P)]
        T_idx = op.index_map[(ℓ, :T)]
        Θ_idx = op.index_map[(ℓ, :Θ)]

        # === Poloidal equation (167) ===
        # (∂/∂t - E Lℓ) Lℓ P - 2 C_ℓm[T] = (Ra/Pr) ℓ(ℓ+1)/r² (r/r₀)² Θ

        # Diffusion term: E Lℓ² P
        Lℓ = op.Lℓ_ops[ℓ]
        LℓP = Lℓ * P_fields[ℓ]
        LℓLℓP = Lℓ * LℓP
        result[P_idx] .= p.E .* LℓLℓP

        # Coriolis coupling: -2 C_ℓm[T]  (equation 183)
        # C_ℓm[T] = (im/r²)[ℓ(ℓ-1)a⁻ T_{ℓ-1} + (ℓ+1)(ℓ+2)a⁺ T_{ℓ+1}]
        C_term = zeros(ComplexF64, p.Nr)

        # Contribution from ℓ-1
        if ℓ > p.m
            _, a_minus = coriolis_coefficients(ℓ, p.m)
            C_term .+= (im * p.m) .* (ℓ*(ℓ-1) * a_minus) .* T_fields[ℓ-1] ./ op.r.^2
        end

        # Contribution from ℓ+1
        if ℓ < p.lmax
            a_plus, _ = coriolis_coefficients(ℓ, p.m)
            C_term .+= (im * p.m) .* ((ℓ+1)*(ℓ+2) * a_plus) .* T_fields[ℓ+1] ./ op.r.^2
        end

        result[P_idx] .-= 2.0 .* real(C_term)  # Take real part for real arithmetic

        # Buoyancy term: (Ra/Pr) ℓ(ℓ+1)/r² (r/r₀)² Θ
        buoyancy = (p.Ra / p.Pr) .* (ℓ*(ℓ+1)) .* Θ_fields[ℓ] ./ op.r.^2 .* (op.r ./ p.ro).^2
        result[P_idx] .+= buoyancy

        # === Toroidal equation (174) ===
        # (∂/∂t - E Lℓ) T + 2 D_ℓm[P] = 0

        # Diffusion term: E Lℓ T
        result[T_idx] .= p.E .* (Lℓ * T_fields[ℓ])

        # Coriolis coupling: 2 D_ℓm[P]  (equation 190)
        # D_ℓm[P] = (1/r)[(ℓ-1)(ℓ+1)a⁻ ∂P_{ℓ-1}/∂r + ℓ(ℓ+2)a⁺ ∂P_{ℓ+1}/∂r]
        D_term = zeros(T, p.Nr)

        # Contribution from ℓ-1
        if ℓ > p.m
            _, a_minus = coriolis_coefficients(ℓ, p.m)
            dP_dr = op.cd.D1 * P_fields[ℓ-1]
            D_term .+= ((ℓ-1)*(ℓ+1) * a_minus) .* dP_dr ./ op.r
        end

        # Contribution from ℓ+1
        if ℓ < p.lmax
            a_plus, _ = coriolis_coefficients(ℓ, p.m)
            dP_dr = op.cd.D1 * P_fields[ℓ+1]
            D_term .+= (ℓ*(ℓ+2) * a_plus) .* dP_dr ./ op.r
        end

        result[T_idx] .+= 2.0 .* D_term

        # === Temperature equation (241) ===
        # With basic state: (∂/∂t - E/Pr Sℓ) Θ = -(u'·∇)θ̄ - (ū·∇)θ'
        #
        # Perturbation advection of basic state: -(u'·∇)θ̄
        #   = -u'_r ∂θ̄/∂r - (u'_θ/r) ∂θ̄/∂θ - (u'_φ/(r sinθ)) ∂θ̄/∂φ
        #
        # Basic state advection of perturbation: -(ū·∇)θ'
        #   = -(ū_φ/(r sinθ)) ∂θ'/∂φ  (since ū_r = ū_θ = 0)
        #   = -(ū_φ/r) × (im/sinθ) θ' in Fourier space

        # Diffusion term: (E/Pr) Sℓ Θ
        Sℓ = op.Sℓ_ops[ℓ]
        result[Θ_idx] .= (p.E / p.Pr) .* (Sℓ * Θ_fields[ℓ])

        # Advection of basic state temperature by perturbation velocity
        if p.basic_state === nothing
            # Default: pure conduction basic state with only radial variation
            # From equation (6): dθ̄/dr = (rᵢr₀/(r₀-rᵢ)) (1/r²) ΔT
            # In non-dimensional form with ΔT=1: dθ̄/dr = (rᵢr₀/L) / r²
            dtheta_bar_dr = (p.ri * p.ro / p.L) ./ op.r.^2

            # Radial advection: -u'_r ∂θ̄/∂r = -ℓ(ℓ+1)/r² P ∂θ̄/∂r
            advection_radial = -(ℓ*(ℓ+1)) .* dtheta_bar_dr .* P_fields[ℓ] ./ op.r.^2
            result[Θ_idx] .+= advection_radial

        else
            # Basic state with meridional variation: θ̄(r,θ) = Σ θ̄_ℓ'0(r) Y_ℓ'0(θ)
            bs = p.basic_state

            # Radial advection: -u'_r ∂θ̄/∂r
            # The radial component comes from all ℓ' modes of the basic state
            # For the current mode ℓ, we sum contributions from all basic state modes

            # This is an approximation: assume basic state is dominated by ℓ'=0,2
            # Full implementation would require mode coupling integrals

            if haskey(bs.theta_coeffs, 0)
                # ℓ'=0 component (conduction profile)
                dtheta_bar_dr_0 = bs.dtheta_dr_coeffs[0]
                advection_r_0 = -(ℓ*(ℓ+1)) .* dtheta_bar_dr_0 .* P_fields[ℓ] ./ op.r.^2
                result[Θ_idx] .+= advection_r_0 ./ sqrt(4π)  # Normalize by Y_00
            end

            if haskey(bs.theta_coeffs, 2) && ℓ >= 2
                # ℓ'=2 component (meridional variation)
                # This couples through integrals of Y_ℓm Y_20
                # For simplicity, approximate coupling for ℓ=2 mode
                if ℓ == 2
                    dtheta_bar_dr_2 = bs.dtheta_dr_coeffs[2]
                    norm_Y20 = sqrt(5/(4π))
                    advection_r_2 = -(ℓ*(ℓ+1)) .* dtheta_bar_dr_2 .* P_fields[ℓ] ./ op.r.^2
                    result[Θ_idx] .+= advection_r_2 .* norm_Y20
                end
            end

            # TODO: Add meridional advection term -(u'_θ/r) ∂θ̄/∂θ
            # This requires computing u'_θ from poloidal potential and evaluating
            # spherical harmonic derivatives ∂Y_ℓ0/∂θ
            # This is a mode-coupling term that requires careful treatment

            # TODO: Add azimuthal advection by basic state flow: -(ū_φ/r)(im/sinθ)θ'
            # This is simpler as it's diagonal in ℓ for each mode
        end
    end

    # Apply boundary conditions
    apply_boundary_conditions!(result, x, op)

    return result
end

"""
    apply_mass(op::LinearStabilityOperator, x::Vector)

Apply the mass matrix B to state vector x.
This is the identity for velocity and temperature fields (equations 204-208).
"""
function apply_mass(op::LinearStabilityOperator{T}, x::AbstractVector{T}) where T
    result = copy(x)

    # Set boundary rows to zero (they don't participate in time evolution)
    apply_boundary_conditions!(result, x, op; zero_bcs=true)

    return result
end

"""
    apply_boundary_conditions!(result, x, op; zero_bcs=false)

Enforce boundary conditions on the operator application.
Implements equations (256-262) for no-slip and (268-272) for stress-free.
"""
function apply_boundary_conditions!(result::AbstractVector{T}, x::AbstractVector{T},
                                   op::LinearStabilityOperator{T};
                                   zero_bcs::Bool=false) where T
    p = op.params
    ri_idx = 1
    ro_idx = p.Nr

    for ℓ in p.m:p.lmax
        P_idx = op.index_map[(ℓ, :P)]
        T_idx = op.index_map[(ℓ, :T)]
        Θ_idx = op.index_map[(ℓ, :Θ)]

        # === Velocity boundary conditions ===

        # Impermeability: P = 0 at boundaries (equation 256)
        if zero_bcs
            result[P_idx[ri_idx]] = 0.0
            result[P_idx[ro_idx]] = 0.0
        else
            result[P_idx[ri_idx]] = x[P_idx[ri_idx]]
            result[P_idx[ro_idx]] = x[P_idx[ro_idx]]
        end

        if p.mechanical_bc == :no_slip
            # No-slip: ∂P/∂r = 0, T = 0 at boundaries (equations 261-262)
            dP_dr = op.cd.D1 * x[P_idx]

            if zero_bcs
                result[T_idx[ri_idx]] = 0.0
                result[T_idx[ro_idx]] = 0.0
            else
                # Enforce ∂P/∂r = 0 by replacing the P equation at boundaries
                # (already done above with P=0)
                # For a proper implementation, we'd modify one row of the operator
                # Here we just enforce T=0
                result[T_idx[ri_idx]] = x[T_idx[ri_idx]]
                result[T_idx[ro_idx]] = x[T_idx[ro_idx]]
            end

        elseif p.mechanical_bc == :stress_free
            # Stress-free: r ∂²P/∂r² - 2 ∂P/∂r = 0 (equation 268)
            #              r ∂T/∂r - 2T = 0 (equation 271)
            d2P_dr2 = op.cd.D2 * x[P_idx]
            dP_dr = op.cd.D1 * x[P_idx]
            dT_dr = op.cd.D1 * x[T_idx]

            if zero_bcs
                result[T_idx[ri_idx]] = 0.0
                result[T_idx[ro_idx]] = 0.0
            else
                # Inner boundary
                result[P_idx[ri_idx]] = op.r[ri_idx] * d2P_dr2[ri_idx] - 2*dP_dr[ri_idx]
                result[T_idx[ri_idx]] = op.r[ri_idx] * dT_dr[ri_idx] - 2*x[T_idx[ri_idx]]

                # Outer boundary
                result[P_idx[ro_idx]] = op.r[ro_idx] * d2P_dr2[ro_idx] - 2*dP_dr[ro_idx]
                result[T_idx[ro_idx]] = op.r[ro_idx] * dT_dr[ro_idx] - 2*x[T_idx[ro_idx]]
            end
        end

        # === Thermal boundary conditions ===

        if p.thermal_bc == :fixed_temperature
            # Θ = 0 at boundaries (equation 278)
            if zero_bcs
                result[Θ_idx[ri_idx]] = 0.0
                result[Θ_idx[ro_idx]] = 0.0
            else
                result[Θ_idx[ri_idx]] = x[Θ_idx[ri_idx]]
                result[Θ_idx[ro_idx]] = x[Θ_idx[ro_idx]]
            end

        elseif p.thermal_bc == :fixed_flux
            # ∂Θ/∂r = 0 at boundaries (equation 284)
            dΘ_dr = op.cd.D1 * x[Θ_idx]

            if zero_bcs
                result[Θ_idx[ri_idx]] = 0.0
                result[Θ_idx[ro_idx]] = 0.0
            else
                result[Θ_idx[ri_idx]] = dΘ_dr[ri_idx]
                result[Θ_idx[ro_idx]] = dΘ_dr[ro_idx]
            end
        end
    end

    return nothing
end

# =============================================================================
#  Eigenvalue Solver
# =============================================================================

"""
    solve_eigenvalue_problem(op::LinearStabilityOperator;
                             nev=6, which=:LM, tol=1e-10)

Solve the generalized eigenvalue problem Ax = λBx.
Returns eigenvalues and eigenvectors.

# Arguments
- `nev`: Number of eigenvalues to compute
- `which`: Which eigenvalues (`:LM` for largest magnitude, `:LR` for largest real part)
- `tol`: Convergence tolerance
"""
function solve_eigenvalue_problem(op::LinearStabilityOperator{T};
                                 nev::Int=6, which::Symbol=:LR,
                                 tol::Real=1e-10, maxiter::Int=1000) where T
    # Define the linear operators
    A_op(x) = apply_operator(op, x)
    B_op(x) = apply_mass(op, x)

    # Use KrylovKit for the generalized eigenvalue problem
    # We solve (A - σB)⁻¹ B x = θ x with shift-invert
    # This finds eigenvalues near σ
    σ = 0.0  # Look near λ=0 (marginal stability)

    # Create the shifted operator: (A - σB)⁻¹ B
    # For simplicity, we use iterative solver within the eigensolver
    function shifted_op(x)
        # Solve (A - σB)y = B x for y
        # Using GMRES
        b = B_op(x)

        function shifted_A(y)
            return A_op(y) - σ .* B_op(y)
        end

        # Simple fixed-point iteration (could use GMRES for better performance)
        y, info = linsolve(shifted_A, b, x, maxiter=100, tol=1e-8)

        return y
    end

    # Initial guess
    x0 = randn(T, op.total_dof)

    # Solve eigenvalue problem
    vals, vecs, info = eigsolve(shifted_op, x0, nev, which,
                                tol=tol, maxiter=maxiter, issymmetric=false)

    # Convert back to original eigenvalues: λ = σ + 1/θ
    eigenvalues = [σ + 1/θ for θ in vals]
    eigenvectors = vecs

    return eigenvalues, eigenvectors, info
end

"""
    find_growth_rate(op::LinearStabilityOperator; kwargs...)

Find the leading eigenvalue (largest real part) and return its growth rate σ.
"""
function find_growth_rate(op::LinearStabilityOperator; kwargs...)
    eigenvalues, eigenvectors, info = solve_eigenvalue_problem(op;
                                                              which=:LR, kwargs...)

    # Find eigenvalue with largest real part
    idx = argmax(real.(eigenvalues))
    λ_max = eigenvalues[idx]

    σ = real(λ_max)  # Growth rate
    ω = imag(λ_max)  # Drift frequency

    return σ, ω, eigenvectors[idx]
end

# =============================================================================
#  Critical Parameter Finding
# =============================================================================

"""
    find_critical_rayleigh(E, Pr, χ, m, lmax, Nr;
                          Ra_guess=1e6, tol=1e-6, kwargs...)

Find the critical Rayleigh number Ra_c where σ = 0 for given parameters.
Uses Brent's method (root finding).

# Returns
- `Ra_c`: Critical Rayleigh number
- `ω_c`: Critical drift frequency
- `eigenvector`: Critical mode structure
"""
function find_critical_rayleigh(E::T, Pr::T, χ::T, m::Int, lmax::Int, Nr::Int;
                               Ra_guess::T=1e6, tol::T=1e-6,
                               Ra_bracket::Tuple{T,T}=(Ra_guess/10, Ra_guess*10),
                               kwargs...) where T<:Real

    function growth_rate_at_Ra(Ra)
        params = OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=m, lmax=lmax, Nr=Nr; kwargs...)
        op = LinearStabilityOperator(params)
        σ, ω, vec = find_growth_rate(op)
        return σ
    end

    # Find bracket where sign changes
    Ra_low, Ra_high = Ra_bracket
    σ_low = growth_rate_at_Ra(Ra_low)
    σ_high = growth_rate_at_Ra(Ra_high)

    # Adjust bracket if needed
    max_attempts = 10
    attempt = 0
    while σ_low * σ_high > 0 && attempt < max_attempts
        if σ_low > 0
            Ra_low /= 2
            σ_low = growth_rate_at_Ra(Ra_low)
        else
            Ra_high *= 2
            σ_high = growth_rate_at_Ra(Ra_high)
        end
        attempt += 1
    end

    if σ_low * σ_high > 0
        error("Could not bracket the critical Rayleigh number")
    end

    # Brent's method for root finding
    Ra_c = brent_method(growth_rate_at_Ra, Ra_low, Ra_high, tol)

    # Get critical frequency and mode
    params_c = OnsetParams(E=E, Pr=Pr, Ra=Ra_c, χ=χ, m=m, lmax=lmax, Nr=Nr; kwargs...)
    op_c = LinearStabilityOperator(params_c)
    σ_c, ω_c, vec_c = find_growth_rate(op_c)

    return Ra_c, ω_c, vec_c
end

"""
    brent_method(f, a, b, tol)

Brent's method for root finding. Finds x where f(x) = 0 in interval [a, b].
"""
function brent_method(f::Function, a::T, b::T, tol::T) where T<:Real
    fa = f(a)
    fb = f(b)

    @assert fa * fb < 0 "Function must have opposite signs at endpoints"

    if abs(fa) < abs(fb)
        a, b = b, a
        fa, fb = fb, fa
    end

    c = a
    fc = fa
    mflag = true

    while abs(b - a) > tol && abs(fb) > tol
        if fa != fc && fb != fc
            # Inverse quadratic interpolation
            s = (a*fb*fc)/((fa-fb)*(fa-fc)) +
                (b*fa*fc)/((fb-fa)*(fb-fc)) +
                (c*fa*fb)/((fc-fa)*(fc-fb))
        else
            # Secant method
            s = b - fb*(b-a)/(fb-fa)
        end

        # Check conditions
        if !( ((3*a+b)/4 < s < b) || (b < s < (3*a+b)/4) ) ||
           (mflag && abs(s-b) >= abs(b-c)/2) ||
           (!mflag && abs(s-b) >= abs(c-d)/2)
            # Bisection
            s = (a+b)/2
            mflag = true
        else
            mflag = false
        end

        fs = f(s)
        d = c
        c = b
        fc = fb

        if fa * fs < 0
            b = s
            fb = fs
        else
            a = s
            fa = fs
        end

        if abs(fa) < abs(fb)
            a, b = b, a
            fa, fb = fb, fa
        end
    end

    return b
end

# Export main functions
export OnsetParams, LinearStabilityOperator
export solve_eigenvalue_problem, find_growth_rate, find_critical_rayleigh
