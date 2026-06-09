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

# `sh_grid` is a pure function of (lmax, mmax, T) and its result is read-only, so
# memoize it: the self-consistent basic-state Picard loop calls `vecsh_advection`
# once per iteration and would otherwise rebuild the Gauss-Legendre nodes and the
# associated-Legendre tables from scratch every time.
const _SH_GRID_CACHE = Dict{Tuple{Int,Int,DataType}, Any}()

function sh_grid(lmax::Int, mmax::Int, ::Type{T}=Float64) where {T<:Real}
    return get!(_SH_GRID_CACHE, (lmax, mmax, T)) do
        _build_sh_grid(lmax, mmax, T)
    end::SHGrid{T}
end

function _build_sh_grid(lmax::Int, mmax::Int, ::Type{T}) where {T<:Real}
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
basis function `Yf` (default `_sh_Y`; pass `_sh_dYθ` etc. for derivative fields).

For the three built-in basis functions the per-mode normalization, θ-table row and
azimuthal factor are hoisted out of the inner `(j,k)` loop (the `Dict` lookups
`g.N[am]`/`g.P[am]` and the `cos/sin` azimuthal factors are otherwise recomputed
for every grid point). The arithmetic order is preserved, so the result is
bit-identical; any other `Yf` falls back to the per-element path."""
function sh_synthesize(coeffs::AbstractDict{Tuple{Int,Int},T}, g::SHGrid{T};
                       Yf=_sh_Y) where {T}
    Nθ = _sh_Nθ(g); Nφ = _sh_Nφ(g)
    f = zeros(T, Nθ, Nφ)
    kind = Yf === _sh_Y ? 1 : Yf === _sh_dYθ ? 2 : Yf === _sh_dYφ_over_sin ? 3 : 0
    if kind == 0   # generic fallback, unchanged semantics
        for ((ℓ, m), a) in coeffs
            (abs(m) > g.mmax || ℓ > g.lmax) && continue
            @inbounds for j in 1:Nθ, k in 1:Nφ
                f[j, k] += a * Yf(g, ℓ, m, j, k)
            end
        end
        return f
    end
    φf = Vector{T}(undef, Nφ)
    θf = Vector{T}(undef, Nθ)
    for ((ℓ, m), a) in coeffs
        am = abs(m)
        (am > g.mmax || ℓ > g.lmax || ℓ < am) && continue
        kind == 3 && am == 0 && continue          # (1/sinθ)∂φ Ȳ_ℓ0 ≡ 0
        Nam = g.N[am]; Pam = g.P[am]
        Nlm = Nam[ℓ - am + 1]; row = ℓ - am + 1
        if kind == 3
            @inbounds for k in 1:Nφ; φf[k] = _sh_dφfac(g, m, k); end
            @inbounds for j in 1:Nθ; θf[j] = Pam[row, j] / _sh_sinθ(g, j); end
        elseif kind == 2
            @inbounds for k in 1:Nφ; φf[k] = _sh_φfac(g, m, k); end
            @inbounds for j in 1:Nθ; θf[j] = _sh_dPdθ(g, ℓ, am, j); end
        else  # kind == 1
            @inbounds for k in 1:Nφ; φf[k] = _sh_φfac(g, m, k); end
            @inbounds for j in 1:Nθ; θf[j] = Pam[row, j]; end
        end
        @inbounds for j in 1:Nθ, k in 1:Nφ
            f[j, k] += a * (Nlm * θf[j] * φf[k])
        end
    end
    f
end

"""Analyze (project) a scalar grid field onto real-orthonormal SH coefficients.
Per-mode normalization/table/azimuthal factor hoisted out of the inner loop
(bit-identical to the per-element `_sh_Y` form)."""
function sh_analyze(f::AbstractMatrix{T}, g::SHGrid{T}) where {T}
    Nθ = _sh_Nθ(g); Nφ = _sh_Nφ(g)
    coeffs = Dict{Tuple{Int,Int},T}()
    dφ = T(2) * T(π) / Nφ
    φf = Vector{T}(undef, Nφ)
    for m in -g.mmax:g.mmax
        am = abs(m)
        Nam = g.N[am]; Pam = g.P[am]
        @inbounds for k in 1:Nφ; φf[k] = _sh_φfac(g, m, k); end
        for ℓ in am:g.lmax
            Nlm = Nam[ℓ - am + 1]; row = ℓ - am + 1
            acc = zero(T)
            @inbounds for j in 1:Nθ, k in 1:Nφ
                acc += f[j, k] * (Nlm * Pam[row, j] * φf[k]) * g.w[j] * dφ
            end
            coeffs[(ℓ, m)] = acc
        end
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
    Nθ = _sh_Nθ(g); Nφ = _sh_Nφ(g)
    div = Dict{Tuple{Int,Int},T}()
    dφ = T(2) * T(π) / Nφ
    φf = Vector{T}(undef, Nφ); dφf = Vector{T}(undef, Nφ)
    dPθ = Vector{T}(undef, Nθ); Pos = Vector{T}(undef, Nθ)
    for m in -g.mmax:g.mmax
        am = abs(m)
        Nam = g.N[am]; Pam = g.P[am]
        @inbounds for k in 1:Nφ
            φf[k] = _sh_φfac(g, m, k); dφf[k] = _sh_dφfac(g, m, k)
        end
        for ℓ in max(1, am):g.lmax
            Nlm = Nam[ℓ - am + 1]; row = ℓ - am + 1
            @inbounds for j in 1:Nθ
                dPθ[j] = _sh_dPdθ(g, ℓ, am, j)
                Pos[j] = am == 0 ? zero(T) : Pam[row, j] / _sh_sinθ(g, j)
            end
            acc = zero(T)
            @inbounds for j in 1:Nθ, k in 1:Nφ
                acc += (Vθ[j, k] * (Nlm * dPθ[j] * φf[k]) +
                        Vφ[j, k] * (Nlm * Pos[j] * dφf[k])) * g.w[j] * dφ
            end
            s_ℓm = acc / sqrt(T(ℓ * (ℓ + 1)))          # spheroidal coefficient
            div[(ℓ, m)] = -sqrt(T(ℓ * (ℓ + 1))) * s_ℓm  # = -ℓ(ℓ+1)/√(ℓ(ℓ+1)) · s
        end
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

"""
    _sh_nf_to_orth_factor(ℓ, m, T) -> T

Coefficient rescale from the basic-state "no-factorial" SH normalization
(`√((2ℓ+1)/4π·[2 if m≠0])`, used by the thermal-wind/construction subsystem) to
the orthonormal full-`N_ℓm` normalization used by `sh_*`/`vecsh_advection`.
Since ‖Y^{no-factorial}_ℓm‖² = (ℓ+|m|)!/(ℓ-|m|)!, a coefficient transforms as
`a_orth = a_nf · √((ℓ+|m|)!/(ℓ-|m|)!)`. Returns 1 for m=0 (the conventions agree).
"""
function _sh_nf_to_orth_factor(ℓ::Int, m::Int, ::Type{T}) where {T<:Real}
    am = abs(m)
    am == 0 && return one(T)
    ratio = one(T)                       # ∏_{k=ℓ-am+1}^{ℓ+am} k = (ℓ+am)!/(ℓ-am)!
    for k in (ℓ - am + 1):(ℓ + am)
        ratio *= T(k)
    end
    return sqrt(ratio)
end

"""Rescale a coeff dict from no-factorial → orthonormal (`dir=+1`) or back (`dir=-1`)."""
function _sh_rescale(coeffs::AbstractDict{Tuple{Int,Int},Vector{T}}, dir::Int) where {T<:Real}
    out = Dict{Tuple{Int,Int},Vector{T}}()
    for ((ℓ, m), v) in coeffs
        f = _sh_nf_to_orth_factor(ℓ, m, T)
        out[(ℓ, m)] = dir > 0 ? v .* f : v ./ f
    end
    out
end
