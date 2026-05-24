# =============================================================================
#  Real-orthonormal spherical-harmonic transform (cos+sin, ±m) and the
#  vector-harmonic horizontal divergence.
#
#  Foundation for a correct nonaxisymmetric basic-state advection
#  ū·∇T̄ = ∇·(ūT̄) (incompressible). Computing the horizontal divergence in the
#  vector-harmonic basis avoids the aliasing that a scalar ∂_θ term-split incurs
#  (∂_θ of a scalar is not band-limited). Validated by manufactured-solution
#  tests (see test/sh_transform.jl).
#
#  Convention: real orthonormal SH with full N_ℓm,
#     m>0:  Ȳ_ℓm  = √2 N_ℓ|m| P_ℓ|m|(cosθ) cos(mφ)
#     m<0:  Ȳ_ℓm  = √2 N_ℓ|m| P_ℓ|m|(cosθ) sin(|m|φ)
#     m=0:  Ȳ_ℓ0  =    N_ℓ0  P_ℓ0(cosθ)
#  with ∫ Ȳ_ℓm Ȳ_ℓ'm' dΩ = δ_{ℓℓ'} δ_{mm'},  N_ℓm = √((2ℓ+1)/4π · (ℓ-m)!/(ℓ+m)!).
#
#  NOT yet wired into the basic-state solvers (that needs the basic-state storage
#  extended from cos-only to full ±m). Provided as the validated building block.
# =============================================================================

"""Quadrature grid + precomputed associated-Legendre / normalization tables."""
struct SHGrid{T<:Real}
    lmax::Int
    mmax::Int
    μ::Vector{T}              # Gauss-Legendre nodes (cosθ)
    w::Vector{T}              # Gauss-Legendre weights
    φ::Vector{T}              # equispaced azimuthal nodes
    P::Dict{Int,Matrix{T}}    # |m| → P_ℓ|m| table (|m| up to mmax+1 for dθ)
    N::Dict{Int,Vector{T}}    # |m| → N_ℓ|m|
end

function sh_grid(lmax::Int, mmax::Int, ::Type{T}=Float64) where {T<:Real}
    Nθ = 3 * lmax + 6                      # oversampled for product dealiasing
    Nφ = max(3 * mmax + 4, 6)              # ≥ product (2mmax) + test (mmax) azimuthal degree
    μ64, w64 = _gauss_legendre_nodes(Nθ)
    μ = T.(μ64); w = T.(w64)
    φ = T[2 * T(π) * (k - 1) / Nφ for k in 1:Nφ]
    amax = min(mmax + 1, lmax)
    P = Dict{Int,Matrix{T}}(am => _associated_legendre_table(am, lmax, μ) for am in 0:amax)
    N = Dict{Int,Vector{T}}(am => _normalization_table(T, am, lmax) for am in 0:mmax)
    SHGrid{T}(lmax, mmax, μ, w, φ, P, N)
end

_sh_sinθ(g::SHGrid{T}, j::Int) where {T} = sqrt(one(T) - g.μ[j]^2)
_sh_Nθ(g::SHGrid) = length(g.μ)
_sh_Nφ(g::SHGrid) = length(g.φ)

# azimuthal factor and its φ-derivative
function _sh_φfac(g::SHGrid{T}, m::Int, k::Int) where {T}
    m == 0 && return one(T)
    return m > 0 ? sqrt(T(2)) * cos(m * g.φ[k]) : sqrt(T(2)) * sin(-m * g.φ[k])
end
function _sh_dφfac(g::SHGrid{T}, m::Int, k::Int) where {T}
    m == 0 && return zero(T)
    return m > 0 ? sqrt(T(2)) * (-m) * sin(m * g.φ[k]) :
                   sqrt(T(2)) * (-m) * cos(-m * g.φ[k])
end

"""Real orthonormal SH value Ȳ_ℓm at grid node (j,k)."""
function _sh_Y(g::SHGrid{T}, ℓ::Int, m::Int, j::Int, k::Int) where {T}
    am = abs(m)
    (ℓ < am || ℓ > g.lmax || am > g.mmax) && return zero(T)
    g.N[am][ℓ - am + 1] * g.P[am][ℓ - am + 1, j] * _sh_φfac(g, m, k)
end

# pole-safe dP̃_ℓ^am/dθ = ½[P̃^{am+1} - (ℓ+am)(ℓ-am+1) P̃^{am-1}]  (Condon-Shortley)
function _sh_dPdθ(g::SHGrid{T}, ℓ::Int, am::Int, j::Int) where {T}
    ℓ == 0 && return zero(T)
    Pp = (am + 1 <= ℓ && haskey(g.P, am + 1)) ? g.P[am + 1][ℓ - (am + 1) + 1, j] : zero(T)
    Pm = am == 0 ? -g.P[1][ℓ - 1 + 1, j] / (ℓ * (ℓ + 1)) : g.P[am - 1][ℓ - (am - 1) + 1, j]
    T(0.5) * (Pp - T((ℓ + am) * (ℓ - am + 1)) * Pm)
end

"""∂θ Ȳ_ℓm at grid node (j,k)."""
function _sh_dYθ(g::SHGrid{T}, ℓ::Int, m::Int, j::Int, k::Int) where {T}
    am = abs(m)
    (ℓ < am || ℓ > g.lmax || am > g.mmax) && return zero(T)
    g.N[am][ℓ - am + 1] * _sh_dPdθ(g, ℓ, am, j) * _sh_φfac(g, m, k)
end

"""(1/sinθ) ∂φ Ȳ_ℓm — pole-safe (P_ℓ^m ∝ sin^m so /sinθ is clean at interior nodes)."""
function _sh_dYφ_over_sin(g::SHGrid{T}, ℓ::Int, m::Int, j::Int, k::Int) where {T}
    am = abs(m)
    (ℓ < am || am == 0 || ℓ > g.lmax || am > g.mmax) && return zero(T)
    g.N[am][ℓ - am + 1] * (g.P[am][ℓ - am + 1, j] / _sh_sinθ(g, j)) * _sh_dφfac(g, m, k)
end

"""Synthesize a scalar field on the grid from coeffs `Dict{(ℓ,m),value}` using
basis function `Yf` (default `_sh_Y`; pass `_sh_dYθ` etc. for derivative fields)."""
function sh_synthesize(coeffs::AbstractDict{Tuple{Int,Int},T}, g::SHGrid{T};
                       Yf=_sh_Y) where {T}
    f = zeros(T, _sh_Nθ(g), _sh_Nφ(g))
    for ((ℓ, m), a) in coeffs
        (abs(m) > g.mmax || ℓ > g.lmax) && continue
        @inbounds for j in 1:_sh_Nθ(g), k in 1:_sh_Nφ(g)
            f[j, k] += a * Yf(g, ℓ, m, j, k)
        end
    end
    f
end

"""Analyze (project) a scalar grid field onto real-orthonormal SH coefficients."""
function sh_analyze(f::AbstractMatrix{T}, g::SHGrid{T}) where {T}
    coeffs = Dict{Tuple{Int,Int},T}()
    dφ = T(2) * T(π) / _sh_Nφ(g)
    for m in -g.mmax:g.mmax, ℓ in abs(m):g.lmax
        acc = zero(T)
        @inbounds for j in 1:_sh_Nθ(g), k in 1:_sh_Nφ(g)
            acc += f[j, k] * _sh_Y(g, ℓ, m, j, k) * g.w[j] * dφ
        end
        coeffs[(ℓ, m)] = acc
    end
    coeffs
end

"""
    sh_horizontal_divergence(Vθ, Vφ, g) -> Dict{(ℓ,m),value}

Coefficients of the horizontal divergence ∇_h·V_h of the tangent field with grid
components (Vθ, Vφ). Extracts the spheroidal part s_ℓm = (1/√(ℓ(ℓ+1))) ∫ V_h·∇_hȲ_ℓm dΩ
(the toroidal part is divergence-free) and returns ∇_h·V_h coeffs = -√(ℓ(ℓ+1)) s_ℓm.
"""
function sh_horizontal_divergence(Vθ::AbstractMatrix{T}, Vφ::AbstractMatrix{T},
                                  g::SHGrid{T}) where {T}
    div = Dict{Tuple{Int,Int},T}()
    dφ = T(2) * T(π) / _sh_Nφ(g)
    for m in -g.mmax:g.mmax, ℓ in max(1, abs(m)):g.lmax
        acc = zero(T)
        @inbounds for j in 1:_sh_Nθ(g), k in 1:_sh_Nφ(g)
            acc += (Vθ[j, k] * _sh_dYθ(g, ℓ, m, j, k) +
                    Vφ[j, k] * _sh_dYφ_over_sin(g, ℓ, m, j, k)) * g.w[j] * dφ
        end
        s_ℓm = acc / sqrt(T(ℓ * (ℓ + 1)))          # spheroidal coefficient
        div[(ℓ, m)] = -sqrt(T(ℓ * (ℓ + 1))) * s_ℓm  # = -ℓ(ℓ+1)/√(ℓ(ℓ+1)) · s
    end
    div
end

"""
    vecsh_advection(theta, dtheta_dr, ur, dur_dr, utheta, uphi, lmax, mmax, r) -> Dict

Correct nonaxisymmetric advection forcing ū·∇T̄ = ∇·(ūT̄) (assumes incompressible
ū, as for a basic state), computed in the vector-harmonic basis (aliasing-free).
All coefficient dicts are `Dict{(ℓ,m), Vector}` over the radial grid `r`, in the
real-orthonormal SH convention (cos for m>0, sin for m<0). Returns forcing in the
same representation.

Per radius:  ∇·(ūT̄) = (1/r²)∂_r(r² u_r T̄) + (1/r) ∇_h·(T̄ u_h)
           = (2/r)·SH(u_r T̄) + SH(∂_r(u_r T̄)) + (1/r)·∇_h·(T̄ u_θ, T̄ u_φ)
with ∂_r(u_r T̄) = (∂_r u_r) T̄ + u_r (∂_r T̄). Assembled from machine-precision-
validated primitives (`sh_synthesize`/`sh_analyze` round-trip, `sh_horizontal_divergence`).
"""
function vecsh_advection(theta::AbstractDict{Tuple{Int,Int},Vector{T}},
                         dtheta_dr::AbstractDict{Tuple{Int,Int},Vector{T}},
                         ur::AbstractDict{Tuple{Int,Int},Vector{T}},
                         dur_dr::AbstractDict{Tuple{Int,Int},Vector{T}},
                         utheta::AbstractDict{Tuple{Int,Int},Vector{T}},
                         uphi::AbstractDict{Tuple{Int,Int},Vector{T}},
                         lmax::Int, mmax::Int, r::Vector{T}) where {T<:Real}
    g = sh_grid(lmax, mmax, T)
    Nr = length(r)
    forcing = Dict{Tuple{Int,Int},Vector{T}}()
    for m in -mmax:mmax, ℓ in abs(m):lmax
        forcing[(ℓ, m)] = zeros(T, Nr)
    end
    sliceat(d, i) = Dict{Tuple{Int,Int},T}(k => v[i] for (k, v) in d)
    for i in 1:Nr
        Tg  = sh_synthesize(sliceat(theta, i), g)
        dTr = sh_synthesize(sliceat(dtheta_dr, i), g)
        Urg = sh_synthesize(sliceat(ur, i), g)
        dUr = sh_synthesize(sliceat(dur_dr, i), g)
        Uθg = sh_synthesize(sliceat(utheta, i), g)
        Uφg = sh_synthesize(sliceat(uphi, i), g)
        Cr   = sh_analyze(Urg .* Tg, g)                       # SH(u_r T̄)
        CdVr = sh_analyze(dUr .* Tg .+ Urg .* dTr, g)         # SH(∂_r(u_r T̄))
        hdiv = sh_horizontal_divergence(Uθg .* Tg, Uφg .* Tg, g)
        ri = r[i]
        for m in -mmax:mmax, ℓ in abs(m):lmax
            forcing[(ℓ, m)][i] = (T(2) / ri) * get(Cr, (ℓ, m), zero(T)) +
                                 get(CdVr, (ℓ, m), zero(T)) +
                                 (one(T) / ri) * get(hdiv, (ℓ, m), zero(T))
        end
    end
    forcing
end
