# =============================================================================
#  Spherical Harmonic Utilities
#
#  Tools for working with Schmidt semi-normalized spherical harmonics using SHTnsKit:
#  - Derivatives: ∂Y_ℓm/∂θ, ∂Y_ℓm/∂φ
#  - Gaunt coefficients: ∫ Y_ℓ₁m₁ Y_ℓ₂m₂ Y_ℓ₃m₃ dΩ
#  - Mode coupling for advection terms
# =============================================================================

using LinearAlgebra
using SHTnsKit

"""
    theta_derivative_coupling(ℓ1::Int, m1::Int, ℓ2::Int, m2::Int)

Compute the coupling coefficient for ∂Y_ℓ₂m₂/∂θ expanded in Y_ℓ₁m₁ basis.

The derivative of a spherical harmonic can be expressed as:
    ∂Y_ℓm/∂θ = c⁺_ℓm Y_{ℓ+1,m} + c⁻_ℓm Y_{ℓ-1,m}

where the coefficients involve associated Legendre recursion relations.

For Schmidt semi-normalized harmonics:
    ∂Y_ℓm/∂θ = -(ℓ+1) a⁺_ℓm Y_{ℓ+1,m} + ℓ a⁻_ℓm Y_{ℓ-1,m}

where a⁺_ℓm and a⁻_ℓm are the ladder coefficients.

Arguments:
- `ℓ1, m1`: Indices of the test function Y_ℓ₁m₁
- `ℓ2, m2`: Indices of the harmonic being differentiated Y_ℓ₂m₂

Returns:
- Coupling coefficient for ∫ Y_ℓ₁m₁ (∂Y_ℓ₂m₂/∂θ) dΩ
- Zero if m1 ≠ m2 (azimuthal decoupling)
"""
function theta_derivative_coupling(ℓ1::Int, m1::Int, ℓ2::Int, m2::Int)
    # Azimuthal modes must match (no φ dependence in ∂/∂θ)
    if m1 != m2
        return 0.0
    end

    m = m1  # = m2

    # Ladder coefficients for Schmidt semi-normalized harmonics
    # a⁺_ℓm = √[(ℓ+1)² - m²] / √[(2ℓ+1)(2ℓ+3)]
    # a⁻_ℓm = √[ℓ² - m²] / √[(2ℓ-1)(2ℓ+1)]

    function a_plus(ℓ::Int, m::Int)
        if ℓ < abs(m)
            return 0.0
        end
        num = (ℓ+1)^2 - m^2
        den = (2ℓ+1) * (2ℓ+3)
        return sqrt(num / den)
    end

    function a_minus(ℓ::Int, m::Int)
        if ℓ <= abs(m)
            return 0.0
        end
        num = ℓ^2 - m^2
        den = (2ℓ-1) * (2ℓ+1)
        return sqrt(num / den)
    end

    # ∂Y_ℓ₂m/∂θ = -(ℓ₂+1) a⁺_ℓ₂m Y_{ℓ₂+1,m} + ℓ₂ a⁻_ℓ₂m Y_{ℓ₂-1,m}

    if ℓ1 == ℓ2 + 1
        # Contribution from Y_{ℓ₂+1,m} term
        # ∫ Y_ℓ₁m (∂Y_ℓ₂m/∂θ) dΩ = -(ℓ₂+1) a⁺_ℓ₂m × δ_{ℓ₁,ℓ₂+1}
        return -(ℓ2 + 1) * a_plus(ℓ2, m)

    elseif ℓ1 == ℓ2 - 1
        # Contribution from Y_{ℓ₂-1,m} term
        # ∫ Y_ℓ₁m (∂Y_ℓ₂m/∂θ) dΩ = ℓ₂ a⁻_ℓ₂m × δ_{ℓ₁,ℓ₂-1}
        return ℓ2 * a_minus(ℓ2, m)

    else
        # No coupling for |ℓ1 - ℓ2| > 1
        return 0.0
    end
end


"""
    gaunt_coefficient(ℓ1::Int, m1::Int, ℓ2::Int, m2::Int, ℓ3::Int, m3::Int)

Compute the Gaunt coefficient (Wigner 3j-related):
    G(ℓ₁m₁, ℓ₂m₂, ℓ₃m₃) = ∫ Y_ℓ₁m₁ Y_ℓ₂m₂ Y_ℓ₃m₃ dΩ

This integral appears in mode coupling terms like (ū·∇)θ' and (u'·∇)θ̄.

Selection rules:
- m₁ + m₂ + m₃ = 0 (azimuthal selection rule)
- |ℓ₁ - ℓ₂| ≤ ℓ₃ ≤ ℓ₁ + ℓ₂ (triangle inequality)
- ℓ₁ + ℓ₂ + ℓ₃ must be even (parity)

For Schmidt semi-normalized harmonics, this involves Wigner 3j symbols.

NOTE: This is a SIMPLIFIED approximation for diagonal/near-diagonal coupling.
Full implementation requires:
- Wigner 3j symbol calculation
- Factorial computations
- Careful normalization

Arguments:
- (ℓ₁,m₁), (ℓ₂,m₂), (ℓ₃,m₃): Spherical harmonic indices

Returns:
- Gaunt coefficient (simplified/approximate)
"""
function gaunt_coefficient(ℓ1::Int, m1::Int, ℓ2::Int, m2::Int, ℓ3::Int, m3::Int)
    # Selection rule: m₁ + m₂ + m₃ = 0
    if m1 + m2 + m3 != 0
        return 0.0
    end

    # Triangle inequality: |ℓ₁ - ℓ₂| ≤ ℓ₃ ≤ ℓ₁ + ℓ₂
    if ℓ3 < abs(ℓ1 - ℓ2) || ℓ3 > ℓ1 + ℓ2
        return 0.0
    end

    # Parity: ℓ₁ + ℓ₂ + ℓ₃ must be even
    if (ℓ1 + ℓ2 + ℓ3) % 2 != 0
        return 0.0
    end

    # SIMPLIFIED: Diagonal approximation for common cases
    # Full implementation would use Wigner 3j symbols

    # Case 1: All same mode (ℓ₁=ℓ₂=ℓ₃, m₁=m₂=-m₃)
    if ℓ1 == ℓ2 && ℓ2 == ℓ3 && m1 == m2 && m3 == -m1 - m2
        # This gives a simple diagonal coefficient
        norm = sqrt((2ℓ1 + 1) / (4π))
        return norm^3 * sqrt(4π / (2ℓ1 + 1))  # Simplified
    end

    # Case 2: ℓ₂ = 0 (coupling with axisymmetric mode)
    if ℓ2 == 0 && m2 == 0
        if ℓ1 == ℓ3 && m1 == m3
            # Y_ℓm × Y_00 = Y_ℓm / √(4π)
            return 1.0 / sqrt(4π)
        end
    end

    # For other cases, return 0 (approximate)
    # TODO: Implement full Wigner 3j calculation for rigorous treatment
    return 0.0
end


"""
    compute_utheta_from_poloidal(P_ℓm::Vector{T}, ℓ::Int, m::Int,
                                 r::Vector{T}, D1::Matrix{T}) where T

Compute meridional velocity u_θ from poloidal potential P.

From the poloidal-toroidal decomposition:
    u_θ = (1/r) ∂P/∂θ
        = (1/r) Σ_ℓm (∂P_ℓm/∂r) (∂Y_ℓm/∂θ)

In spectral form for mode (ℓ,m):
    u_{θ,ℓm} = (1/r) (∂P_ℓm/∂r) × (coefficient from ∂Y_ℓm/∂θ)

Arguments:
- `P_ℓm`: Radial profile of poloidal potential for mode (ℓ,m)
- `ℓ, m`: Spherical harmonic indices
- `r`: Radial grid
- `D1`: First derivative matrix (Chebyshev)

Returns:
- Vector of radial coefficients for u_θ contribution from this mode
"""
function compute_utheta_from_poloidal(P_ℓm::Vector{T}, ℓ::Int, m::Int,
                                      r::Vector{T}, D1::AbstractMatrix{T}) where T
    # u_θ component involves ∂P/∂r and ∂Y_ℓm/∂θ
    # The angular part gives coupling coefficients
    # Here we return the radial part: (1/r) ∂P_ℓm/∂r

    dP_dr = D1 * P_ℓm
    u_theta_radial = dP_dr ./ r

    return u_theta_radial
end


"""
    meridional_advection_coupling(ℓ_pert::Int, m_pert::Int,
                                  ℓ_bs::Int, m_bs::Int,
                                  ℓ_result::Int)

Compute mode coupling coefficient for meridional advection term:
    -(u'_{θ,ℓ_pert,m_pert}/r) × ∂θ̄_{ℓ_bs,m_bs}/∂θ

This couples to result mode ℓ_result through spherical harmonic products.

The full expression involves:
    ∫ Y_{ℓ_result,m} × [(1/r)(∂P_ℓ_pert/∂r)] × [∂Y_{ℓ_bs,m_bs}/∂θ] dΩ

Returns:
- Coupling coefficient (radial functions multiply separately)
- Zero if selection rules violated
"""
function meridional_advection_coupling(ℓ_pert::Int, m_pert::Int,
                                       ℓ_bs::Int, m_bs::Int,
                                       ℓ_result::Int)
    # Result azimuthal mode from selection rule
    m_result = m_pert + m_bs

    # The ∂Y_{ℓ_bs,m_bs}/∂θ couples to Y_{ℓ_bs±1,m_bs}
    # Then we need to couple with Y_{ℓ_pert,m_pert} to get Y_{ℓ_result,m_result}

    # This requires computing:
    # Σ_{ℓ'} [∂θ̄_{ℓ_bs}/∂θ coupling] × [Gaunt(ℓ_pert, ℓ', ℓ_result)]

    coupling = 0.0

    # ∂Y_{ℓ_bs,m_bs}/∂θ has components at ℓ_bs ± 1
    for ℓ_temp in [ℓ_bs - 1, ℓ_bs + 1]
        if ℓ_temp < abs(m_bs) || ℓ_temp < 0
            continue
        end

        # Coefficient from ∂Y/∂θ
        deriv_coeff = theta_derivative_coupling(ℓ_temp, m_bs, ℓ_bs, m_bs)

        if abs(deriv_coeff) < 1e-14
            continue
        end

        # Gaunt coefficient for Y_{ℓ_pert,m_pert} × Y_{ℓ_temp,m_bs} → Y_{ℓ_result,m_result}
        gaunt = gaunt_coefficient(ℓ_pert, m_pert, ℓ_temp, m_bs, ℓ_result, m_result)

        coupling += deriv_coeff * gaunt
    end

    return coupling
end


"""
    azimuthal_advection_coefficient(ℓ::Int, m::Int, m_bs::Int)

Compute coefficient for azimuthal advection by basic state zonal flow:
    -(ū_φ/r) × (im/sinθ) × θ'

In spectral form, for basic state mode (ℓ_bs, m_bs) and perturbation mode (ℓ,m):
    Couples perturbation m to m ± m_bs

The angular integral gives:
    ∫ Y_ℓm × (1/sinθ) × Y_{ℓ_bs,m_bs} × (im'/sinθ) dΩ

where m' is the perturbation azimuthal number.

Arguments:
- `ℓ, m`: Perturbation mode indices
- `m_bs`: Basic state azimuthal mode

Returns:
- Coupling coefficient for this mode interaction
"""
function azimuthal_advection_coefficient(ℓ::Int, m::Int, m_bs::Int)
    # Simplified: For axisymmetric basic state (m_bs=0), this is diagonal
    if m_bs == 0
        # -(ū_φ,ℓ_bs,0/r) × (im/sinθ) × θ'_ℓm
        # The 1/sinθ factors combine with Y_ℓm evaluation
        # For Schmidt harmonics, this gives a coefficient proportional to m
        return Float64(m)
    end

    # For non-axisymmetric basic state, mode coupling occurs
    # This requires full Gaunt coefficient calculation
    # TODO: Implement for tri-global case

    return 0.0
end
