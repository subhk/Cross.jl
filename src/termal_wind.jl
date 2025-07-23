# =============================================================================
#  Thermal‑wind helper for an arbitrary latitudinal temperature profile
# =============================================================================
"""
    build_thermal_wind(fθ::AbstractVector, r::AbstractVector;
                       gα_2Ω::Float64 = 1.0,
                       Dθ::AbstractMatrix,
                       m::Int, Ek::Float64,
                       r_i::Float64)

Given

* `fθ[k] = f(θ_k)` – the *latitude‑only* temperature anomaly
  (same Gauss–Legendre nodes used elsewhere),
* Chebyshev radial nodes `r`,
* derivative matrix `Dθ`  (`∂/∂θ` on the θ grid),

compute and return the three sparse diagonal matrices

that drop into the linear operator.  All dimensional prefactors
`g α /(2Ω)` are bundled in `gα_2Ω` (set to 1 if you already non‑dimensionalised).
"""
function build_thermal_wind(fθ::AbstractVector,
                            r::AbstractVector;
                            gα_2Ω::Float64 = 1.0,
                            Dθ::AbstractMatrix,
                            m::Int,
                            Ek::Float64,
                            r_i::Float64)

    N_θ = length(fθ)
    N_r = length(r)

    # meridional derivative  f'(θ)
    df_dθ = Dθ * fθ                # size N_θ

    # Thermal‑wind: ∂r Ū = (gα/2Ω)(df/dθ)/r
    # integrate radially so that Ū(r_i,θ)=0   →  Ū(r,θ)=gα_2Ω * df_dθ * ln(r/r_i)
    ln_rr0 = log.(r ./ r_i)                     # length N_r
    Ubar   = (ln_rr0 .* df_dθ') .* gα_2Ω        # (N_r×N_θ) outer product
    dU_dr  = (df_dθ' .* gα_2Ω) ./ r             # by definition

    # flatten in (r,θ) lexicographic order  (θ fastest as in main script)
    Uvec   = vec(Ubar)
    dUvec  = vec(dU_dr)
    dfvec  = repeat(df_dθ', outer=N_r)          # ∂θ f  on tensor grid

    # helpers for sinθ and r*sinθ
    sinθ   = sin.(collect(acos.(fθ * 0 .+ cos.(fθ))))  # fθ already at θ grid
    rs     = repeat(r, inner=N_θ)
    sint   = repeat(sinθ', outer=N_r)

    im_m      = im * m
    U_m = spdiagm(0 => (-Uvec .* im_m) ./ (rs .* sint))   # −Ū ∂φ
    S_r = spdiagm(0 => -dUvec)                            # −u_r ∂rŪ
    S_θ = spdiagm(0 => -dfvec)                            # −u_θ ∂θT̄

    return U_m, S_r, S_θ
end

"""
Example:
U_m, S_r, S_θ = build_thermal_wind(fθ, r;
                                Dθ=Dθ, m=m, Ek=Ek,
                                gα_2Ω=1.0,  # already scaled
                                r_i=r_i)
"""