# =============================================================================
#  Reconstruction of velocity and temperature fields from spectral coefficients
# =============================================================================

using LinearAlgebra
using SHTnsKit

import ..Cross: ChebyshevDiffn

# -----------------------------------------------------------------------------#
#  Internal helpers                                                            #
# -----------------------------------------------------------------------------#

@inline function _mode_length(lmax::Int, m::Int)
    return lmax - m + 1
end

function _extract_mode!(buffer::AbstractVector{ComplexF64},
                        coeffs::AbstractArray{<:Number,3},
                        ir::Int,
                        m::Int,
                        lmax::Int)
    @assert length(buffer) == _mode_length(lmax, m)
    for (idx, ℓ) in enumerate(m:lmax)
        buffer[idx] = complex(coeffs[ir, ℓ + 1, m + 1])
    end
    return buffer
end

function _default_radial_derivative(r::AbstractVector{<:Real})
    nr = length(r)
    domain = (minimum(r), maximum(r))
    cd = ChebyshevDiffn(nr, collect(domain), 1)
    if maximum(abs.(cd.x .- r)) > 1e-10
        @warn "Provided radii do not coincide with Chebyshev collocation points; radial derivatives may be less accurate."
    end
    return cd.D1
end

function _radial_derivatives(coeffs::AbstractArray{<:Number,3}, Dr::AbstractMatrix)
    nr, _, _ = size(coeffs)
    result = Array{ComplexF64,3}(undef, size(coeffs))
    tmp = Vector{ComplexF64}(undef, nr)
    for l in axes(coeffs, 2), m in axes(coeffs, 3)
        mul!(tmp, Dr, complex.(view(coeffs, :, l, m)))
        @inbounds result[:, l, m] .= tmp
    end
    return result
end

function _ensure_complex_array(data::AbstractArray{<:Number,3})
    out = Array{ComplexF64,3}(undef, size(data))
    @inbounds for idx in eachindex(data)
        out[idx] = complex(data[idx])
    end
    return out
end

function _phi_phases(cfg::SHTConfig, m::Int)
    return exp.(im * m .* cfg.φ)
end

# -----------------------------------------------------------------------------#
#  Core synthesis routines                                                     #
# -----------------------------------------------------------------------------#

"""
    velocity_fields_from_poloidal_toroidal(cfg, r, poloidal, toroidal;
                                           Dr=nothing, domain=nothing,
                                           real_output=true)

Synthesize the velocity components `(u_r, u_θ, u_φ)` on the Gauss–Legendre grid
defined by `cfg` from the spectral poloidal (`poloidal`) and toroidal (`toroidal`)
coefficients. The arrays must have shape `(Nr, lmax+1, mmax+1)` where `Nr` is the
number of radial collocation points and the second/third dimensions store the
Chebyshev-slice spherical harmonic coefficients for each `(ℓ, m)`.

* `cfg::SHTConfig` – spherical harmonic configuration (use `create_gauss_config`)
* `r::AbstractVector` – radial positions corresponding to the Chebyshev grid
* `poloidal`, `toroidal` – complex or real spectral coefficients
* `Dr` – optional precomputed first-derivative matrix in radius (defaults to a
         Chebyshev differentiation matrix inferred from `r`)
* `domain` – optional tuple `(ri, ro)` used when building the default derivative
* `real_output` – when `true`, returns real-valued fields assuming the physical
                  solution is the real part of the spectral expansion

Returns `u_r, u_θ, u_φ` arrays of size `(Nr, cfg.nlat, cfg.nlon)`.
"""
function velocity_fields_from_poloidal_toroidal(cfg::SHTConfig,
                                                r::AbstractVector{<:Real},
                                                poloidal::AbstractArray{<:Number,3},
                                                toroidal::AbstractArray{<:Number,3};
                                                Dr::Union{Nothing,AbstractMatrix}=nothing,
                                                domain::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
                                                real_output::Bool=true)
    nr, lp1, mp1 = size(poloidal)
    lp1 == cfg.lmax + 1 || throw(DimensionMismatch("poloidal second dimension must be lmax+1"))
    mp1 == cfg.mmax + 1 || throw(DimensionMismatch("poloidal third dimension must be mmax+1"))
    size(toroidal) == size(poloidal) || throw(DimensionMismatch("toroidal array must match poloidal dimensions"))
    length(r) == nr || throw(DimensionMismatch("length(r) must equal number of radial levels"))

    Dr_used = Dr
    if Dr_used === nothing
        if domain !== nothing
            cd = ChebyshevDiffn(nr, collect(domain), 1)
            Dr_used = cd.D1
            maximum(abs.(cd.x .- r)) < 1e-10 || @warn "Chebyshev nodes derived from `domain` do not perfectly match provided radii."
        else
            Dr_used = _default_radial_derivative(r)
        end
    end

    poloidal_c = _ensure_complex_array(poloidal)
    toroidal_c = _ensure_complex_array(toroidal)
    dP = _radial_derivatives(poloidal_c, Dr_used)

    nlat, nlon = cfg.nlat, cfg.nlon
    out_type = real_output ? Float64 : ComplexF64
    ur = zeros(out_type, nr, nlat, nlon)
    uθ = zeros(out_type, nr, nlat, nlon)
    uφ = zeros(out_type, nr, nlat, nlon)

    Sl = Vector{ComplexF64}(undef, cfg.lmax + 1)
    Tl = similar(Sl)
    Ql = similar(Sl)

    for m in 0:cfg.mmax
        phase = _phi_phases(cfg, m)
        len_mode = _mode_length(cfg.lmax, m)
        resize!(Sl, len_mode)
        resize!(Tl, len_mode)
        resize!(Ql, len_mode)

        for ir in 1:nr
            r_i = r[ir]
            r_inv = 1.0 / r_i
            r_inv2 = r_inv^2
            _extract_mode!(Sl, dP, ir, m, cfg.lmax)
            _extract_mode!(Tl, toroidal_c, ir, m, cfg.lmax)
            _extract_mode!(Ql, poloidal_c, ir, m, cfg.lmax)

            @inbounds begin
                for idx in eachindex(Sl)
                    ℓ = m + idx - 1
                    Sl[idx] *= r_inv
                    Tl[idx] *= -r_inv
                    Ql[idx] *= ℓ * (ℓ + 1) * r_inv2
                end
            end

            Vt_m, Vp_m = SHsphtor_to_spat_ml(cfg, m, Sl, Tl, cfg.lmax)
            Vr_m = SH_to_spat_ml(cfg, m, Ql, cfg.lmax)

            uθ_slice = view(uθ, ir, :, :)
            uφ_slice = view(uφ, ir, :, :)
            ur_slice = view(ur, ir, :, :)

            if real_output && m == 0
                @inbounds for j in 1:nlon
                    phase_j = phase[j]
                    for i in 1:nlat
                        valθ = Vt_m[i] * phase_j
                        valφ = Vp_m[i] * phase_j
                        valr = Vr_m[i] * phase_j
                        uθ_slice[i, j] = real(valθ)
                        uφ_slice[i, j] = real(valφ)
                        ur_slice[i, j] = real(valr)
                    end
                end
            else
                for j in 1:nlon
                    phase_j = phase[j]
                    for i in 1:nlat
                        valθ = Vt_m[i] * phase_j
                        valφ = Vp_m[i] * phase_j
                        valr = Vr_m[i] * phase_j
                        if real_output
                            uθ_slice[i, j] += 2 * real(valθ)
                            uφ_slice[i, j] += 2 * real(valφ)
                            ur_slice[i, j] += 2 * real(valr)
                        else
                            uθ_slice[i, j] += valθ
                            uφ_slice[i, j] += valφ
                            ur_slice[i, j] += valr
                        end
                    end
                end
            end
        end
    end

    return ur, uθ, uφ
end

"""
    temperature_field_from_coefficients(cfg, temperature_coeffs;
                                        real_output=true)

Synthesize the perturbation temperature on the spherical grid given the
spherical harmonic coefficients stored as `(Nr, lmax+1, mmax+1)`. Returns an
array of size `(Nr, cfg.nlat, cfg.nlon)`.
"""
function temperature_field_from_coefficients(cfg::SHTConfig,
                                             temperature_coeffs::AbstractArray{<:Number,3};
                                             real_output::Bool=true)
    nr, lp1, mp1 = size(temperature_coeffs)
    lp1 == cfg.lmax + 1 || throw(DimensionMismatch("temperature coefficients must have lmax+1 rows"))
    mp1 == cfg.mmax + 1 || throw(DimensionMismatch("temperature coefficients must have mmax+1 columns"))

    temp_c = _ensure_complex_array(temperature_coeffs)
    nlat, nlon = cfg.nlat, cfg.nlon
    out = zeros(real_output ? Float64 : ComplexF64, nr, nlat, nlon)

    for m in 0:cfg.mmax
        phase = _phi_phases(cfg, m)
        len_mode = _mode_length(cfg.lmax, m)
        coeff_mode = Vector{ComplexF64}(undef, len_mode)

        for ir in 1:nr
            _extract_mode!(coeff_mode, temp_c, ir, m, cfg.lmax)
            Vr_m = SH_to_spat_ml(cfg, m, coeff_mode, cfg.lmax)
            slice = view(out, ir, :, :)
            if real_output && m == 0
                for j in 1:nlon
                    phase_j = phase[j]
                    for i in 1:nlat
                        slice[i, j] = real(Vr_m[i] * phase_j)
                    end
                end
            else
                for j in 1:nlon
                    phase_j = phase[j]
                    for i in 1:nlat
                        val = Vr_m[i] * phase_j
                        if real_output
                            slice[i, j] += 2 * real(val)
                        else
                            slice[i, j] += val
                        end
                    end
                end
            end
        end
    end

    return out
end

"""
    fields_from_coefficients(cfg, r, poloidal, toroidal, temperature;
                             Dr=nothing, domain=nothing, real_output=true)

Convenience routine that simultaneously reconstructs velocity and temperature
fields from spectral coefficients. Returns `(u_r, u_θ, u_φ, θ')`, where the last
entry is `nothing` if `temperature === nothing`.
"""
function fields_from_coefficients(cfg::SHTConfig,
                                  r::AbstractVector{<:Real},
                                  poloidal::AbstractArray{<:Number,3},
                                  toroidal::AbstractArray{<:Number,3},
                                  temperature::Union{Nothing,AbstractArray{<:Number,3}}=nothing;
                                  Dr::Union{Nothing,AbstractMatrix}=nothing,
                                  domain::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
                                  real_output::Bool=true)
    ur, uθ, uφ = velocity_fields_from_poloidal_toroidal(cfg, r, poloidal, toroidal;
                                                        Dr=Dr, domain=domain,
                                                        real_output=real_output)
    θfield = temperature === nothing ? nothing :
             temperature_field_from_coefficients(cfg, temperature;
                                                 real_output=real_output)
    return ur, uθ, uφ, θfield
end

# -----------------------------------------------------------------------------#
#  Backwards compatibility helper                                              #
# -----------------------------------------------------------------------------#

"""
    potentials_to_velocity(P, T; Dr, Dθ, Lθ, r, sintheta, m)

Deprecated compatibility wrapper. Computes `(u_r, u_θ, u_φ)` on the meridional
grid for a single azimuthal mode using the analytic formulas. Prefer
`velocity_fields_from_poloidal_toroidal` for new code.
"""
function potentials_to_velocity(P::AbstractMatrix,
                                T::AbstractMatrix;
                                Dr,
                                Dθ,
                                Lθ,
                                r::AbstractVector,
                                sintheta::AbstractVector,
                                m::Int)
    Nr, Nθ = size(P)
    size(T) == size(P) || throw(DimensionMismatch("P and T must have same size"))
    @assert size(Dr, 1) == Nr
    @assert size(Dθ, 1) == Nθ
    @assert length(r) == Nr
    @assert length(sintheta) == Nθ

    inv_r = 1.0 ./ r
    inv_r2 = inv_r .^ 2
    inv_sinθ = 1.0 ./ sintheta

    dθ_P = P * Dθ'
    dθ_T = T * Dθ'
    lap_ang_P = P * Lθ'
    dP_dr = Dr * P

    ur = -lap_ang_P .* inv_r2
    uθ = dP_dr * Dθ'
    uθ .= uθ .* inv_r .* (ones(Nr) * ones(Nθ)')
    uθ .+= (im * m) .* T .* (inv_r * inv_sinθ')

    uφ = (im * m) .* dP_dr .* (inv_r * inv_sinθ')
    uφ .-= dθ_T .* (inv_r * ones(1, Nθ))

    return ur, uθ, uφ
end
