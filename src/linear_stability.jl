module LinearStability

using LinearAlgebra
using SHTnsKit
using KrylovKit

import ..Cross: ChebyshevDiffn

export ShellParams,
       MeridionalOperator,
       setup_operator,
       leading_modes,
       apply_operator,
       apply_mass

"""
    ShellParams(; m, E, Pr, Ra, ri, ro, lmax, Nr)

Container for the dimensionless control parameters entering the linear
stability problem described by Equations (10)–(19) of
`docs/Onset_convection.pdf`. The fields correspond to:

  • `m`   – fixed azimuthal wavenumber of the perturbation (Equation 12)
  • `E`   – Ekman number multiplying viscous terms in Equation (10)
  • `Pr`  – Prandtl number coupling momentum and heat Equations (10)–(11)
  • `Ra`  – Rayleigh number prefactor of the buoyancy term in Equation (10)
  • `ri`, `ro` – nondimensional radii of the spherical shell boundaries
  • `lmax` – spherical harmonic truncation in latitude
  • `Nr`  – number of Chebyshev collocation points across the shell

All floating-point inputs are promoted to `Float64` so the downstream
operators have a consistent element type.
"""
struct ShellParams{T<:Real}
    m::Int
    E::T
    Pr::T
    Ra::T
    ri::T
    ro::T
    lmax::Int
    Nr::Int
end

function ShellParams(; m::Int,
                        E::Real,
                        Pr::Real,
                        Ra::Real,
                        ri::Real,
                        ro::Real,
                        lmax::Int,
                        Nr::Int)
    return ShellParams{Float64}(m, float(E), float(Pr), float(Ra), float(ri), float(ro), lmax, Nr)
end

# -----------------------------------------------------------------------------
#  Latitudinal operators from SHTnsKit Gauss grid
# -----------------------------------------------------------------------------

function build_latitude_operators(cfg::SHTConfig, m::Int, lmax::Int)
    prepare_plm_tables!(cfg)
    nθ = cfg.nlat
    ℓvals = m:lmax
    nℓ = length(ℓvals)

    Ptab = cfg.plm_tables[m+1]
    dPtab = cfg.dplm_tables[m+1]
    x = cfg.x
    sinθ = sqrt.(1 .- x.^2)

    Y = zeros(Float64, nθ, nℓ)
    dY = zeros(Float64, nθ, nℓ)

    for (col, ℓ) in enumerate(ℓvals)
        Y[:, col] .= Ptab[ℓ+1, :]
        dY[:, col] .= -sinθ .* dPtab[ℓ+1, :]
    end

    W = Diagonal(cfg.wlat)
    gram = Y' * W * Y
    Ydag = gram \ (Y' * W)  # pseudo-inverse respecting quadrature weights

    ℓop = ℓvals .* (ℓvals .+ 1)
    Dθ = dY * Ydag
    Lθ = Y * (-Diagonal(Float64.(ℓop))) * Ydag
    return Dθ, Lθ
end

# -----------------------------------------------------------------------------
#  Meridional 2-D operator (r,θ) for a fixed azimuthal wavenumber m
# -----------------------------------------------------------------------------

struct MeridionalOperator
    params      :: ShellParams
    Nr          :: Int
    Nθ          :: Int
    r           :: Vector{Float64}
    θ           :: Vector{Float64}
    sinθ        :: Vector{Float64}
    cosθ        :: Vector{Float64}
    sinθ_row    :: Matrix{Float64}
    cosθ_row    :: Matrix{Float64}
    inv_sinθ_row:: Matrix{Float64}
    inv_sinθ2_row:: Matrix{Float64}
    r_mat       :: Matrix{Float64}
    r2_mat      :: Matrix{Float64}
    inv_r       :: Matrix{Float64}
    inv_r2      :: Matrix{Float64}
    inv_r_sinθ  :: Matrix{Float64}
    inv_r_sinθ2 :: Matrix{Float64}
    r_over_ro   :: Matrix{Float64}
    dT_dr       :: Matrix{Float64}
    im_m        :: ComplexF64
    m2          :: Float64
    Dr          :: Matrix{Float64}
    D2          :: Matrix{Float64}
    Dθ          :: Matrix{Float64}
    Lθ          :: Matrix{Float64}
    gauge_r     :: Int
    gauge_θ     :: Int
end

function setup_operator(params::ShellParams; nθ::Int = params.lmax + 1)
    cd = ChebyshevDiffn(params.Nr, [params.ri, params.ro], 2)
    r = cd.x
    Dr = cd.D₁
    D2 = cd.D₂

    cfg = create_gauss_config(params.lmax, nθ; mmax=params.m, mres=1, nlon=max(2*params.m + 1, 4))
    enable_plm_tables!(cfg)
    x = cfg.x
    θ = acos.(clamp.(x, -1.0, 1.0))
    sinθ = sqrt.(1 .- x.^2)
    cosθ = x

    Dθ, Lθ = build_latitude_operators(cfg, params.m, params.lmax)

    Nr = params.Nr
    Nθ = nθ

    r_mat = repeat(reshape(r, Nr, 1), 1, Nθ)
    r2_mat = r_mat .^ 2
    sinθ_row = reshape(sinθ, 1, Nθ)
    cosθ_row = reshape(cosθ, 1, Nθ)
    inv_sinθ_row = 1.0 ./ sinθ_row
    inv_sinθ2_row = inv_sinθ_row .^ 2
    inv_r = 1.0 ./ r_mat
    inv_r2 = 1.0 ./ r2_mat
    inv_r_sinθ = inv_r .* inv_sinθ_row
    inv_r_sinθ2 = inv_r .* inv_sinθ2_row
    r_over_ro = r_mat ./ params.ro
    dT_profile = -(params.ri * params.ro) / (params.ro - params.ri) .* (1.0 ./ r.^2)
    dT_dr = repeat(reshape(dT_profile, Nr, 1), 1, Nθ)

    gauge_r = Nr
    gauge_θ = cld(Nθ, 2)

    return MeridionalOperator(params,
                              Nr,
                              Nθ,
                              r,
                              θ,
                              sinθ,
                              cosθ,
                              sinθ_row,
                              cosθ_row,
                              inv_sinθ_row,
                              inv_sinθ2_row,
                              r_mat,
                              r2_mat,
                              inv_r,
                              inv_r2,
                              inv_r_sinθ,
                              inv_r_sinθ2,
                              r_over_ro,
                              dT_dr,
                              im * params.m,
                              float(params.m^2),
                              Dr,
                              D2,
                              Dθ,
                              Lθ,
                              gauge_r,
                              gauge_θ)
end

# -----------------------------------------------------------------------------
#  Helpers to reshape unknown vectors into field blocks
# -----------------------------------------------------------------------------

@inline function split_fields(op::MeridionalOperator, x::AbstractVector{ComplexF64})
    N = op.Nr * op.Nθ
    ur = reshape(view(x, 1:N), op.Nr, op.Nθ)
    uθ = reshape(view(x, N+1:2N), op.Nr, op.Nθ)
    uφ = reshape(view(x, 2N+1:3N), op.Nr, op.Nθ)
    Θ  = reshape(view(x, 3N+1:4N), op.Nr, op.Nθ)
    p  = reshape(view(x, 4N+1:5N), op.Nr, op.Nθ)
    return ur, uθ, uφ, Θ, p
end

@inline function pack_fields(res_r, res_θ, res_φ, res_T, res_div)
    return vcat(vec(res_r), vec(res_θ), vec(res_φ), vec(res_T), vec(res_div))
end

# -----------------------------------------------------------------------------
#  Core operator evaluations
# -----------------------------------------------------------------------------

function apply_operator(op::MeridionalOperator, x::AbstractVector{ComplexF64})
    ur, uθ, uφ, Θ, p = split_fields(op, x)

    # Derivatives
    dθ_ur = ur * op.Dθ'
    dθ_uθ = uθ * op.Dθ'
    dθ_uφ = uφ * op.Dθ'
    dθ_Θ = Θ * op.Dθ'
    dθ_p = p  * op.Dθ'

    ∂r_ur = op.Dr * ur
    ∂r_uθ = op.Dr * uθ
    ∂r_uφ = op.Dr * uφ
    ∂r_p  = op.Dr * p

    # Curl
    dθ_sinθ_uφ = op.cosθ_row .* uφ .+ op.sinθ_row .* dθ_uφ
    ruφ = uφ .* op.r_mat
    ∂r_ruφ = op.Dr * ruφ
    ruθ = uθ .* op.r_mat
    ∂r_ruθ = op.Dr * ruθ

    vort_r = op.inv_r_sinθ .* (dθ_sinθ_uφ .- op.im_m .* uθ)
    vort_θ = op.inv_r .* (op.im_m .* ur .- ∂r_ruφ)
    vort_φ = op.inv_r .* (∂r_ruθ .- dθ_ur)

    # Curl of vorticity (→ Laplacian)
    dθ_vort_r = vort_r * op.Dθ'
    dθ_vort_φ = vort_φ * op.Dθ'
    dθ_sinθ_vort_φ = op.cosθ_row .* vort_φ .+ op.sinθ_row .* dθ_vort_φ
    rvortφ = vort_φ .* op.r_mat
    ∂r_rvortφ = op.Dr * rvortφ
    rvortθ = vort_θ .* op.r_mat
    ∂r_rvortθ = op.Dr * rvortθ

    curlcurl_r = op.inv_r_sinθ .* (dθ_sinθ_vort_φ .- op.im_m .* vort_θ)
    curlcurl_θ = op.inv_r .* (op.im_m .* vort_r .- ∂r_rvortφ)
    curlcurl_φ = op.inv_r .* (∂r_rvortθ .- dθ_vort_r)

    lap_u_r = -curlcurl_r
    lap_u_θ = -curlcurl_θ
    lap_u_φ = -curlcurl_φ

    # Pressure gradient
    grad_p_r = ∂r_p
    grad_p_θ = op.inv_r .* dθ_p
    grad_p_φ = op.im_m .* p .* op.inv_r_sinθ

    # Coriolis term (2 Ω × u with Ω = ẑ)
    zcross_r = -op.sinθ_row .* uφ
    zcross_θ = -op.cosθ_row .* uφ
    zcross_φ =  op.cosθ_row .* uθ .+ op.sinθ_row .* ur

    # Buoyancy contribution
    buoy_r = (op.params.Ra / op.params.Pr) .* op.r_over_ro .* Θ

    # Scalar Laplacian acting on Θ
    dΘ_dr  = op.Dr * Θ
    d2Θ_dr2 = op.D2 * Θ
    lat_raw = (op.sinθ_row .* dθ_Θ) * op.Dθ'
    lat_term = lat_raw .* (op.inv_r2 .* op.inv_sinθ_row)
    phi_term = -op.m2 .* Θ .* op.inv_r2 .* op.inv_sinθ2_row
    lap_Θ = d2Θ_dr2 .+ 2 .* op.inv_r .* dΘ_dr .+ lat_term .+ phi_term

    # Momentum residuals (no-slip rows overwritten below)
    res_r = -grad_p_r .- 2 .* zcross_r .+ buoy_r .+ op.params.E .* lap_u_r
    res_θ = -grad_p_θ .- 2 .* zcross_θ .+ op.params.E .* lap_u_θ
    res_φ = -grad_p_φ .- 2 .* zcross_φ .+ op.params.E .* lap_u_φ

    # Temperature equation
    res_T = -(op.dT_dr .* ur) .+ (op.params.E / op.params.Pr) .* lap_Θ

    # Divergence constraint with pressure gauge substitution
    term_r = (op.Dr * (ur .* op.r2_mat)) .* op.inv_r2
    term_θ = op.inv_r_sinθ .* ((op.sinθ_row .* uθ) * op.Dθ')
    term_φ = op.im_m .* uφ .* op.inv_r_sinθ
    res_div = term_r .+ term_θ .+ term_φ
    res_div[op.gauge_r, op.gauge_θ] = p[op.gauge_r, op.gauge_θ]

    # Boundary conditions: no-slip and fixed temperature
    res_r[1, :] .= ur[1, :]
    res_r[end, :] .= ur[end, :]
    res_θ[1, :] .= uθ[1, :]
    res_θ[end, :] .= uθ[end, :]
    res_φ[1, :] .= uφ[1, :]
    res_φ[end, :] .= uφ[end, :]
    res_T[1, :] .= Θ[1, :]
    res_T[end, :] .= Θ[end, :]
    res_div[1, :] .= 0
    res_div[end, :] .= 0

    return pack_fields(res_r, res_θ, res_φ, res_T, res_div)
end

function apply_mass(op::MeridionalOperator, x::AbstractVector{ComplexF64})
    ur, uθ, uφ, Θ, _ = split_fields(op, x)
    mass_r = copy(ur)
    mass_θ = copy(uθ)
    mass_φ = copy(uφ)
    mass_T = copy(Θ)
    mass_div = zeros(ComplexF64, op.Nr, op.Nθ)

    mass_r[1, :] .= 0
    mass_r[end, :] .= 0
    mass_θ[1, :] .= 0
    mass_θ[end, :] .= 0
    mass_φ[1, :] .= 0
    mass_φ[end, :] .= 0
    mass_T[1, :] .= 0
    mass_T[end, :] .= 0

    return pack_fields(mass_r, mass_θ, mass_φ, mass_T, mass_div)
end

# -----------------------------------------------------------------------------
#  Eigenvalue interface (KrylovKit)
# -----------------------------------------------------------------------------

function leading_modes(params::ShellParams; nθ::Int=params.lmax + 1,
                                           nev::Int=6,
                                           which::Symbol=:LR,
                                           kwargs...)
    op = setup_operator(params; nθ=nθ)
    Ndof = 5 * op.Nr * op.Nθ

    actionA = x -> apply_operator(op, x)
    actionB = x -> apply_mass(op, x)

    v0 = randn(ComplexF64, Ndof)
    vals, vecs, info = eigsolve(actionA, actionB, Ndof, nev, which;
                                v0=v0, kwargs...)
    return vals, vecs, op, info
end

end
