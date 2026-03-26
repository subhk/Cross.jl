# ============================================================================
# Input validation for Cross.jl parameter types
# ============================================================================

"""
    validate_onset_params(params)

Validate onset parameters. Throws `ArgumentError` for invalid values,
emits `@warn` for unusual-but-valid combinations.
Called automatically in the OnsetParams constructor.
"""
function validate_onset_params(params)
    # --- Hard errors ---
    0 < params.χ < 1 || throw(ArgumentError(
        "Radius ratio χ must be in (0,1), got $(params.χ)"))
    params.E > 0 || throw(ArgumentError(
        "Ekman number E must be positive, got $(params.E)"))
    params.Pr > 0 || throw(ArgumentError(
        "Prandtl number Pr must be positive, got $(params.Pr)"))
    params.Ra >= 0 || throw(ArgumentError(
        "Rayleigh number Ra must be non-negative, got $(params.Ra)"))
    params.Nr >= 8 || throw(ArgumentError(
        "Nr must be >= 8 for meaningful resolution, got $(params.Nr)"))
    params.lmax >= 1 || throw(ArgumentError(
        "lmax must be >= 1, got $(params.lmax)"))
    params.m >= 0 || throw(ArgumentError(
        "Azimuthal wavenumber m must be >= 0, got $(params.m)"))
    params.mechanical_bc in (:no_slip, :stress_free) || throw(ArgumentError(
        "mechanical_bc must be :no_slip or :stress_free, got :$(params.mechanical_bc)"))
    params.thermal_bc in (:fixed_temperature, :fixed_flux) || throw(ArgumentError(
        "thermal_bc must be :fixed_temperature or :fixed_flux, got :$(params.thermal_bc)"))
    params.equatorial_symmetry in (:both, :symmetric, :antisymmetric) || throw(ArgumentError(
        "equatorial_symmetry must be :both, :symmetric, or :antisymmetric, got :$(params.equatorial_symmetry)"))

    # --- Warnings ---
    params.Nr < 16 && @warn "Nr=$(params.Nr) is very low — results may be under-resolved"
    params.E > 0.1 && @warn "E=$(params.E) is unusually large — Coriolis effects may be negligible"
    params.E < 1e-8 && @warn "E=$(params.E) is very small — may require high Nr and lmax for convergence"
    params.lmax > 3 * params.Nr && @warn "Angular resolution far exceeds radial: lmax=$(params.lmax) >> Nr=$(params.Nr)"
    params.m > params.lmax && @warn "No modes will be included: m=$(params.m) > lmax=$(params.lmax)"

    return nothing
end

"""
    validate_basic_state_consistency(bs::BasicState, params)

Cross-validate that a BasicState is compatible with the given parameters.
"""
function validate_basic_state_consistency(bs, params)
    bs.Nr == params.Nr || throw(ArgumentError(
        "BasicState Nr=$(bs.Nr) doesn't match params Nr=$(params.Nr)"))
    return nothing
end

"""
    validate_basic_state_3d_consistency(bs::BasicState3D, params)

Cross-validate that a BasicState3D is compatible with the given parameters.
"""
function validate_basic_state_3d_consistency(bs, params)
    bs.Nr == params.Nr || throw(ArgumentError(
        "BasicState3D Nr=$(bs.Nr) doesn't match params Nr=$(params.Nr)"))
    return nothing
end

"""
    validate_biglobal_params(params, basic_state)

Validate biglobal parameters. Applies all onset checks to the base physics
parameters and cross-validates the basic state consistency.
"""
function validate_biglobal_params(params, basic_state)
    validate_onset_params(params)
    validate_basic_state_consistency(basic_state, params)
    return nothing
end

"""
    validate_triglobal_params(params, basic_state, m_range)

Validate triglobal parameters. Applies all onset checks to the base physics
parameters, validates the m_range, and cross-validates the 3D basic state.
"""
function validate_triglobal_params(params, basic_state, m_range)
    validate_onset_params(params)
    validate_basic_state_3d_consistency(basic_state, params)

    isempty(m_range) && throw(ArgumentError(
        "m_range must be non-empty"))
    first(m_range) >= 0 || @warn "m_range starts at $(first(m_range)) — negative m modes included"

    return nothing
end
