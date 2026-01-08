# =============================================================================
#  Thermal-wind helper for an arbitrary latitudinal temperature profile
# =============================================================================

using SparseArrays
"""
    build_thermal_wind(fθ::AbstractVector, r::AbstractVector;
                       gα_2Ω::Float64 = 1.0,
                       Dθ::AbstractMatrix,
                       m::Int,
                       r_i::Float64,
                       r_o::Union{Nothing,Float64}=nothing,
                       sintheta::Union{Nothing,AbstractVector}=nothing)

Given

* `fθ[k] = f(θ_k)` – the latitude-only temperature anomaly
  (same Gauss–Legendre nodes used elsewhere),
* Chebyshev radial nodes `r`,
* derivative matrix `Dθ`  (`∂/∂θ` on the θ grid),

compute and return the three sparse diagonal matrices

that drop into the linear operator.  All dimensional prefactors
`g α /(2Ω)` are bundled in `gα_2Ω` (set to 1 if you already non-dimensionalised).
If using the Cross.jl non-dimensionalization, set `gα_2Ω = Ra * E^2 / (2 * Pr)`.
"""
function build_thermal_wind(fθ::AbstractVector,
                            r::AbstractVector;
                            gα_2Ω::Float64 = 1.0,
                            Dθ::AbstractMatrix,
                            m::Int,
                            r_i::Float64,
                            r_o::Union{Nothing,Float64}=nothing,
                            sintheta::Union{Nothing,AbstractVector}=nothing)

    N_θ = length(fθ)
    N_r = length(r)
    size(Dθ, 1) == N_θ || throw(DimensionMismatch("Dθ must have $N_θ rows"))
    size(Dθ, 2) == N_θ || throw(DimensionMismatch("Dθ must have $N_θ columns"))

    # meridional derivative  f'(θ)
    df_dθ = Dθ * fθ                # size N_θ

    sinθ = sintheta
    if sinθ === nothing
        throw(ArgumentError("sintheta must be provided to build_thermal_wind"))
    end
    length(sinθ) == N_θ || throw(DimensionMismatch("sintheta must have length $N_θ"))
    any(abs.(sinθ) .< eps(Float64)) && throw(ArgumentError("sintheta must be nonzero"))

    r_o_val = r_o === nothing ? maximum(r) : r_o

    # Thermal wind: dU/dr + U/r = -(gα/2Ω) * (1/(r_o sinθ)) * dθ̄/dθ
    rhs = -(gα_2Ω / r_o_val) .* (df_dθ ./ sinθ)   # length N_θ
    r2_minus = r .^ 2 .- r_i^2                    # length N_r
    Ubar = (0.5 .* r2_minus) .* rhs' ./ r         # (N_r x N_θ)
    dU_dr = rhs' .- (Ubar ./ r)                   # by definition

    # flatten in (r,θ) lexicographic order (r fastest: column-major from N_r × N_θ matrix)
    # Index pattern: (r₁,θ₁), (r₂,θ₁), ..., (r_Nr,θ₁), (r₁,θ₂), ...
    Uvec   = vec(Ubar)
    dUvec  = vec(dU_dr)

    # Repeat patterns for r-fastest ordering:
    # - rs: each r value appears once, then repeats for each θ: [r₁,r₂,...,r_Nr, r₁,r₂,...,r_Nr, ...]
    # - sint/dfvec: each θ value repeats N_r times: [s₁,s₁,...,s₁, s₂,s₂,...,s₂, ...]
    rs     = repeat(r, outer=N_θ)              # r values repeated for each θ block
    sint   = repeat(vec(sinθ), inner=N_r)      # sin(θ) repeated N_r times per θ value
    dfvec  = repeat(vec(df_dθ), inner=N_r)     # ∂θf repeated N_r times per θ value

    im_m      = im * m
    U_m = spdiagm(0 => (-Uvec .* im_m) ./ (rs .* sint))   # −Ū ∂φ
    S_r = spdiagm(0 => -dUvec)                            # −u_r ∂rŪ
    S_θ = spdiagm(0 => -(dfvec ./ rs))                    # −u_θ (1/r) ∂θT̄

    return U_m, S_r, S_θ
end

"""
Example:
U_m, S_r, S_θ = build_thermal_wind(fθ, r;
                                Dθ=Dθ, m=m,
                                gα_2Ω=Ra * E^2 / (2 * Pr),
                                r_i=r_i,
                                r_o=r_o,
                                sintheta=sintheta)
"""
