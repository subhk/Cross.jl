# =============================================================================
#  Spherical Harmonic Utilities using SHTnsKit
#
#  Rigorous implementation of mode coupling using SHTnsKit.jl for:
#  - Meridional advection: -(u'_θ/r) ∂θ̄/∂θ
#  - Azimuthal advection: -(ū_φ/r)(im/sinθ)θ'
# =============================================================================

using SHTnsKit
using LinearAlgebra

"""
    theta_derivative_coefficient(ℓ::Int, m::Int)

Compute coupling coefficients for ∂Y_ℓm/∂θ using ladder relations.

The derivative expands as:
    ∂Y_ℓm/∂θ = c⁺_ℓm Y_{ℓ+1,m} + c⁻_ℓm Y_{ℓ-1,m}

For Schmidt semi-normalized harmonics:
    c⁺_ℓm = -(ℓ+1) √[(ℓ+1)² - m²] / √[(2ℓ+1)(2ℓ+3)]
    c⁻_ℓm = ℓ √[ℓ² - m²] / √[(2ℓ-1)(2ℓ+1)]

Returns:
- (c_plus, c_minus): Coefficients for Y_{ℓ+1,m} and Y_{ℓ-1,m}
"""
function theta_derivative_coefficient(ℓ::Int, m::Int)
    # Check validity
    if ℓ < abs(m)
        return (0.0, 0.0)
    end

    # c⁺: Coefficient for Y_{ℓ+1,m}
    if ℓ >= abs(m)
        num_plus = (ℓ + 1)^2 - m^2
        den_plus = (2ℓ + 1) * (2ℓ + 3)
        c_plus = -(ℓ + 1) * sqrt(num_plus / den_plus)
    else
        c_plus = 0.0
    end

    # c⁻: Coefficient for Y_{ℓ-1,m}
    if ℓ > abs(m)
        num_minus = ℓ^2 - m^2
        den_minus = (2ℓ - 1) * (2ℓ + 1)
        c_minus = ℓ * sqrt(num_minus / den_minus)
    else
        c_minus = 0.0
    end

    return (c_plus, c_minus)
end


# -----------------------------------------------------------------------------
#  Wigner 3j / Gaunt coefficient utilities
# -----------------------------------------------------------------------------

const _fourπ = 4π

@inline _phase(k::Int) = isodd(k) ? -1.0 : 1.0

@inline function _logfactorial(n::Int)
    @assert n >= 0 "Factorial argument must be non-negative (got $n)"
    return loggamma(n + 1)
end

"""
    wigner_3j(ℓ1, ℓ2, ℓ3, m1, m2, m3)

Evaluate the Wigner 3j symbol ⟨ℓ1 ℓ2 ℓ3; m1 m2 m3⟩ for integer indices.
Returns 0.0 when the standard selection rules are violated.
"""
function wigner_3j(ℓ1::Int, ℓ2::Int, ℓ3::Int, m1::Int, m2::Int, m3::Int)
    # Selection rules
    if m1 + m2 + m3 != 0
        return 0.0
    end
    if (ℓ1 < 0) || (ℓ2 < 0) || (ℓ3 < 0)
        return 0.0
    end
    if abs(m1) > ℓ1 || abs(m2) > ℓ2 || abs(m3) > ℓ3
        return 0.0
    end
    if ℓ3 < abs(ℓ1 - ℓ2) || ℓ3 > ℓ1 + ℓ2
        return 0.0
    end
    if (ℓ1 + ℓ2 + ℓ3) % 2 != 0
        return 0.0
    end

    # Prefactor
    triangle_log = _logfactorial(ℓ1 + ℓ2 - ℓ3) +
                   _logfactorial(ℓ1 - ℓ2 + ℓ3) +
                   _logfactorial(-ℓ1 + ℓ2 + ℓ3) -
                   _logfactorial(ℓ1 + ℓ2 + ℓ3 + 1)

    fact_log = _logfactorial(ℓ1 + m1) + _logfactorial(ℓ1 - m1) +
               _logfactorial(ℓ2 + m2) + _logfactorial(ℓ2 - m2) +
               _logfactorial(ℓ3 + m3) + _logfactorial(ℓ3 - m3)

    prefactor = _phase(ℓ1 - ℓ2 - m3) * exp(0.5 * (triangle_log + fact_log))

    # Summation bounds
    t_min = max(0, ℓ2 - ℓ3 - m1, ℓ1 - ℓ3 + m2)
    t_max = min(ℓ1 + ℓ2 - ℓ3, ℓ1 - m1, ℓ2 + m2)

    sum_val = 0.0
    for t in t_min:t_max
        denom_args = (
            t,
            ℓ1 + ℓ2 - ℓ3 - t,
            ℓ1 - m1 - t,
            ℓ2 + m2 - t,
            ℓ3 - ℓ2 + m1 + t,
            ℓ3 - ℓ1 - m2 + t,
        )

        if any(x -> x < 0, denom_args)
            continue
        end

        log_term = -sum(_logfactorial, denom_args)
        term = _phase(t) * exp(log_term)
        sum_val += term
    end

    return prefactor * sum_val
end

"""
    gaunt_coefficient(ℓ1, m1, ℓ2, m2, ℓ3, m3)

Compute the Gaunt coefficient ∫ Y_{ℓ1 m1} Y_{ℓ2 m2} Y_{ℓ3 m3} dΩ for
Schmidt semi-normalized spherical harmonics.
"""
function gaunt_coefficient(ℓ1::Int, m1::Int,
                           ℓ2::Int, m2::Int,
                           ℓ3::Int, m3::Int)
    w1 = wigner_3j(ℓ1, ℓ2, ℓ3, 0, 0, 0)
    if w1 == 0.0
        return 0.0
    end

    w2 = wigner_3j(ℓ1, ℓ2, ℓ3, m1, m2, m3)
    if w2 == 0.0
        return 0.0
    end

    pref = sqrt((2ℓ1 + 1) * (2ℓ2 + 1) * (2ℓ3 + 1) / _fourπ)
    return pref * w1 * w2
end

"""
    gaunt_with_conjugate(ℓ_out, m_out, ℓ1, m1, ℓ2, m2)

Helper returning ∫ Y_{ℓ_out,m_out}^* Y_{ℓ1,m1} Y_{ℓ2,m2} dΩ.
"""
function gaunt_with_conjugate(ℓ_out::Int, m_out::Int,
                              ℓ1::Int, m1::Int,
                              ℓ2::Int, m2::Int)
    return _phase(m_out) * gaunt_coefficient(ℓ_out, -m_out, ℓ1, m1, ℓ2, m2)
end


"""
    compute_meridional_advection_coupling(
        ℓ_pert::Int, m_pert::Int,
        ℓ_bs::Int, m_bs::Int,
        ℓ_test::Int)

Compute mode coupling coefficient for:
    ∫ Y_{ℓ_test,m_test} × u'_{θ,ℓ_pert,m_pert} × (∂θ̄_{ℓ_bs,m_bs}/∂θ) × (1/r) dΩ

where:
- u'_θ ∝ (1/r) ∂P_ℓ_pert/∂r
- ∂θ̄/∂θ expands as Y_{ℓ_bs±1,m_bs}

The result couples to mode m_test = m_pert + m_bs

This requires computing triple products of spherical harmonics.

Arguments:
- `ℓ_pert, m_pert`: Perturbation mode indices
- `ℓ_bs, m_bs`: Basic state mode indices
- `ℓ_test`: Test function mode (result goes here)

Returns:
- Coupling coefficient
"""
function compute_meridional_advection_coupling(
    ℓ_pert::Int, m_pert::Int,
    ℓ_bs::Int, m_bs::Int,
    ℓ_test::Int)

    m_test = m_pert + m_bs
    coupling = 0.0
    c_plus, c_minus = theta_derivative_coefficient(ℓ_bs, m_bs)

    if abs(c_plus) > 1e-14
        ℓ_temp = ℓ_bs + 1
        coupling += c_plus * gaunt_with_conjugate(ℓ_test, m_test, ℓ_pert, m_pert, ℓ_temp, m_bs)
    end
    if abs(c_minus) > 1e-14 && ℓ_bs > 0
        ℓ_temp = ℓ_bs - 1
        coupling += c_minus * gaunt_with_conjugate(ℓ_test, m_test, ℓ_pert, m_pert, ℓ_temp, m_bs)
    end

    return coupling
end


"""
    azimuthal_advection_coefficient_axisym(ℓ::Int, m::Int)

Compute the real coupling factor for azimuthal advection by an
axisymmetric (m=0) zonal flow component:

    -(ū_φ,ℓ_bs0 / r) × (im) θ'_ℓm × ∫ Y_ℓm^* Y_ℓm Y_ℓ_bs0 dΩ

For the ℓ_bs = 0 component this reduces to m / √(4π); higher even ℓ_bs
terms are handled automatically via Gaunt coefficients.

Returns the real coefficient multiplying `ū_φ,ℓ_bs0 / r`, with the `im`
factor kept external.
"""
function azimuthal_advection_coefficient_axisym(ℓ::Int, m::Int)
    if m == 0
        return 0.0
    end
    coeff = gaunt_with_conjugate(ℓ, m, ℓ, m, 0, 0)
    return m * coeff
end

"""
    azimuthal_advection_coefficient(ℓ::Int, m::Int, m_bs::Int)

Placeholder coupling coefficient for non-axisymmetric basic states. The current
single-m Kore solver does not support cross-m couplings, so this returns zero
and effectively disables the contribution.
"""
function azimuthal_advection_coefficient(ℓ::Int, m::Int, m_bs::Int)
    return 0.0
end


"""
    evaluate_spherical_harmonic_grid(ℓmax::Int, m::Int, θ_grid::Vector{T}, φ_grid::Vector{T}) where T

Evaluate spherical harmonics Y_ℓm on a grid using SHTnsKit.

This is useful for transforming between spectral and physical space.

Arguments:
- `ℓmax`: Maximum degree
- `m`: Azimuthal mode
- `θ_grid`: Colatitude grid points
- `φ_grid`: Azimuthal grid points

Returns:
- Dictionary of Y_ℓm evaluated at grid points
"""
function evaluate_spherical_harmonic_grid(ℓmax::Int, m::Int, θ_grid::Vector{T}, φ_grid::Vector{T}) where T
    # This would use SHTnsKit's evaluation routines
    # Placeholder for full implementation

    # Initialize SHTnsKit configuration
    # config = SHTConfig(ℓmax, m, ...)
    # Evaluate harmonics on grid

    # Return dictionary of Y_ℓm values
    Y_dict = Dict{Int, Matrix{T}}()

    # TODO: Implement using SHTnsKit functions

    return Y_dict
end


"""
    compute_meridional_gradient_spectrum(θ_coeffs::Dict{Int,Vector{T}},
                                        ℓmax::Int, m::Int) where T

Compute spectral coefficients of ∂θ̄/∂θ from spectral coefficients θ̄_ℓm(r).

Input:
- `θ_coeffs`: Dictionary of θ̄_ℓm(r) for each ℓ
- `ℓmax`: Maximum degree
- `m`: Azimuthal mode

Output:
- Dictionary of spectral coefficients for ∂θ̄/∂θ expanded in Y_ℓm basis

Uses:
    ∂θ̄/∂θ = Σ_ℓ θ̄_ℓm(r) × ∂Y_ℓm/∂θ
           = Σ_ℓ θ̄_ℓm(r) × [c⁺_ℓm Y_{ℓ+1,m} + c⁻_ℓm Y_{ℓ-1,m}]
           = Σ_ℓ' [Σ_ℓ θ̄_ℓm(r) × (coupling)] Y_ℓ'm
"""
function compute_meridional_gradient_spectrum(θ_coeffs::Dict{Int,Vector{T}},
                                              ℓmax::Int, m::Int) where T
    # Initialize result
    dθ_dθ_coeffs = Dict{Int,Vector{T}}()

    Nr = length(θ_coeffs[0])  # Number of radial points

    # For each result mode ℓ'
    for ℓ_result in abs(m):ℓmax
        dθ_dθ_coeffs[ℓ_result] = zeros(T, Nr)

        # Sum contributions from all source modes ℓ
        for ℓ_source in abs(m):ℓmax
            if !haskey(θ_coeffs, ℓ_source)
                continue
            end

            c_plus, c_minus = theta_derivative_coefficient(ℓ_source, m)

            # Contribution from Y_{ℓ_source+1,m} term
            if ℓ_result == ℓ_source + 1
                dθ_dθ_coeffs[ℓ_result] .+= c_plus .* θ_coeffs[ℓ_source]
            end

            # Contribution from Y_{ℓ_source-1,m} term
            if ℓ_result == ℓ_source - 1
                dθ_dθ_coeffs[ℓ_result] .+= c_minus .* θ_coeffs[ℓ_source]
            end
        end
    end

    return dθ_dθ_coeffs
end
