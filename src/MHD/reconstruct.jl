# =============================================================================
#  Physical-space reconstruction of MHD perturbation fields from an eigenvector.
#
#  Native to the MHD spectral (Chebyshev-coefficient) basis: scatter an interior
#  eigenvector back to the full DOF layout, slice the per-(field, ℓ) coefficient
#  blocks, evaluate each Chebyshev series on a radial grid (Clenshaw), then
#  synthesize physical fields on a meridional (r, θ) grid.  The poloidal-toroidal
#  curl mirrors `potentials_to_velocity` exactly (velocity and magnetic field
#  share the same B = ∇×∇×(P r̂) + ∇×(T r̂) form).
# =============================================================================

"""Total (pre-BC) MHD degrees of freedom: all five sections × `(N+1)` radial coeffs."""
function _mhd_total_dof(op::MHDStabilityOperator)
    n_modes = length(op.ll_u) + length(op.ll_v) + length(op.ll_f) +
              length(op.ll_g) + length(op.ll_h)
    return n_modes * (op.params.N + 1)
end

"""
    _mhd_full_vector(evec, op, interior_dofs)

Return a full-length DOF vector. If `evec` already has `_mhd_total_dof(op)`
entries it is returned unchanged. If `evec` matches `length(interior_dofs)`,
its entries are scattered into a zero full vector at `interior_dofs`.
"""
function _mhd_full_vector(evec::AbstractVector{<:Complex},
                          op::MHDStabilityOperator,
                          interior_dofs)
    ndof = _mhd_total_dof(op)
    if length(evec) == ndof
        return Vector{ComplexF64}(evec)
    elseif interior_dofs !== nothing && length(evec) == length(interior_dofs)
        full = zeros(ComplexF64, ndof)
        full[interior_dofs] .= evec
        return full
    else
        error("_mhd_full_vector: eigenvector length $(length(evec)) matches neither " *
              "the full DOF count $ndof nor length(interior_dofs)=" *
              "$(interior_dofs === nothing ? "nothing" : length(interior_dofs)). " *
              "Pass interior_dofs from assemble_mhd_matrices.")
    end
end

"""Slice the `(N+1)` spectral coefficients for `(ℓ, field)` from a full vector."""
function _mhd_field_block(full::AbstractVector{<:Complex},
                          idx_map::Dict{Tuple{Int,Symbol},UnitRange{Int}},
                          field::Symbol, ℓ::Int)
    return @view full[idx_map[(ℓ, field)]]
end

"""
    _mhd_radial_eval(coeffs, ricb, r_grid)

Evaluate a Chebyshev-T coefficient series (`coeffs[n+1]` multiplies `T_n`) at
physical radii `r_grid ∈ [ricb, 1]`, via `x = 2(r-ricb)/(1-ricb) - 1` and
`T_n(x) = cos(n·acos(x))`.
"""
function _mhd_radial_eval(coeffs::AbstractVector{<:Complex},
                          ricb::Real, r_grid::AbstractVector)
    N = length(coeffs) - 1
    out = zeros(ComplexF64, length(r_grid))
    @inbounds for (i, r) in enumerate(r_grid)
        x = 2 * (r - ricb) / (1 - ricb) - 1
        x = clamp(x, -1.0, 1.0)
        ax = acos(x)
        acc = zero(ComplexF64)
        for n in 0:N
            acc += coeffs[n + 1] * cos(n * ax)
        end
        out[i] = acc
    end
    return out
end

"""
Radial reconstruction grid: the Chebyshev collocation nodes of a
`ChebyshevDiffn(Nr, [ricb, 1], 1)`. `_mhd_poltor_to_physical` builds the SAME
`ChebyshevDiffn`, so its `D1` matches these nodes exactly (no grid mismatch).
"""
function _mhd_radial_grid(op::MHDStabilityOperator; Nr::Int = op.params.N + 1)
    return ChebyshevDiffn(Nr, [op.params.ricb, 1.0], 1).x
end

"""
    perturbation_temperature(evec, op::MHDStabilityOperator;
                             Nθ=nothing, Nr=nothing, interior_dofs=nothing, grid=nothing)

Reconstruct the physical perturbation temperature field
`θ(r,θ) = Σ_ℓ h_ℓ(r) Y_ℓ^m(θ)` from an MHD eigenvector (interior or full).
Returns `(θfield, r_grid, grid)`.
"""
function perturbation_temperature(evec::AbstractVector{<:Complex},
                                  op::MHDStabilityOperator;
                                  Nθ::Union{Int,Nothing}=nothing,
                                  Nr::Union{Int,Nothing}=nothing,
                                  interior_dofs=nothing,
                                  grid::Union{MeridionalGrid,Nothing}=nothing)
    m    = op.params.m
    lmax = op.params.lmax
    ricb = op.params.ricb
    full     = _mhd_full_vector(evec, op, interior_dofs)
    idx_map  = _mhd_index_map(op)
    g = grid === nothing ?
        build_meridional_grid(Nθ === nothing ? 2 * lmax : Nθ, m, lmax) : grid
    r_grid = _mhd_radial_grid(op; Nr = Nr === nothing ? op.params.N + 1 : Nr)

    θfield = zeros(ComplexF64, length(r_grid), length(g.θ))
    for l in op.ll_h
        hl = _mhd_radial_eval(_mhd_field_block(full, idx_map, :h, l), ricb, r_grid)
        ylm = g.Ylm[l]
        @inbounds for j in eachindex(g.θ), k in eachindex(r_grid)
            θfield[k, j] += hl[k] * ylm[j]
        end
    end
    return θfield, r_grid, g
end

# Poloidal-toroidal → physical curl, mirroring `potentials_to_velocity`:
#   F_r = -L²P / r²              (L² acting on the synthesized poloidal field)
#   F_θ = (1/r) ∂²P/∂r∂θ + (im·m / (r sinθ)) T
#   F_φ = (im·m / (r sinθ)) ∂P/∂r - (1/r) ∂T/∂θ
# P, T are synthesized poloidal/toroidal potential fields on (r_grid, θ).
function _mhd_poltor_to_physical(full, idx_map, op,
                                 ls_pol, sec_pol::Symbol,
                                 ls_tor, sec_tor::Symbol,
                                 r_grid, g::MeridionalGrid)
    ricb = op.params.ricb
    Nr = length(r_grid); Nθ = length(g.θ)

    # Synthesize potential fields P(r,θ), T(r,θ) on the grid.
    P   = zeros(ComplexF64, Nr, Nθ)
    Tor = zeros(ComplexF64, Nr, Nθ)
    for l in ls_pol
        pl = _mhd_radial_eval(_mhd_field_block(full, idx_map, sec_pol, l), ricb, r_grid)
        ylm = g.Ylm[l]
        @inbounds for j in 1:Nθ, k in 1:Nr
            P[k, j] += pl[k] * ylm[j]
        end
    end
    for l in ls_tor
        tl = _mhd_radial_eval(_mhd_field_block(full, idx_map, sec_tor, l), ricb, r_grid)
        ylm = g.Ylm[l]
        @inbounds for j in 1:Nθ, k in 1:Nr
            Tor[k, j] += tl[k] * ylm[j]
        end
    end

    # Radial derivative on the Chebyshev-Lobatto r_grid.
    cd = ChebyshevDiffn(Nr, [ricb, 1.0], 1)
    Dr = cd.D1

    Fr    = P * transpose(g.Lθ)        # L² P
    dP_dr = Dr * P                      # ∂P/∂r
    Fθ    = dP_dr * transpose(g.Dθ)    # ∂²P/∂r∂θ
    Fφ    = Tor * transpose(g.Dθ)      # ∂T/∂θ

    im_m = ComplexF64(im * op.params.m)   # imaginary unit × m  (im = Base.im here)
    @inbounds for j in 1:Nθ
        inv_sinθ = inv(g.sinθ[j])
        for k in 1:Nr
            inv_r = inv(r_grid[k])
            inv_r_sinθ = inv_r * inv_sinθ
            Fr[k, j] = -Fr[k, j] * inv_r * inv_r
            Fθ[k, j] = Fθ[k, j] * inv_r + im_m * Tor[k, j] * inv_r_sinθ
            Fφ[k, j] = im_m * dP_dr[k, j] * inv_r_sinθ - Fφ[k, j] * inv_r
        end
    end
    return Fr, Fθ, Fφ
end

"""
    perturbation_velocity(evec, op::MHDStabilityOperator; kwargs...)

Reconstruct physical perturbation velocity `(u_r, u_θ, u_φ)` from an MHD
eigenvector (poloidal `:u`, toroidal `:v`). Returns `(ur, uθ, uφ, r_grid, grid)`.
"""
function perturbation_velocity(evec::AbstractVector{<:Complex},
                               op::MHDStabilityOperator;
                               Nθ::Union{Int,Nothing}=nothing,
                               Nr::Union{Int,Nothing}=nothing,
                               interior_dofs=nothing,
                               grid::Union{MeridionalGrid,Nothing}=nothing)
    full     = _mhd_full_vector(evec, op, interior_dofs)
    idx_map  = _mhd_index_map(op)
    g = grid === nothing ?
        build_meridional_grid(Nθ === nothing ? 2 * op.params.lmax : Nθ,
                              op.params.m, op.params.lmax) : grid
    r_grid = _mhd_radial_grid(op; Nr = Nr === nothing ? op.params.N + 1 : Nr)
    Fr, Fθ, Fφ = _mhd_poltor_to_physical(full, idx_map, op,
                                          op.ll_u, :u, op.ll_v, :v, r_grid, g)
    return Fr, Fθ, Fφ, r_grid, g
end

"""
    perturbation_magnetic(evec, op::MHDStabilityOperator; kwargs...)

Reconstruct physical perturbation magnetic field `(B_r, B_θ, B_φ)` from an MHD
eigenvector (poloidal `:f`, toroidal `:g`). Returns `(Br, Bθ, Bφ, r_grid, grid)`.
"""
function perturbation_magnetic(evec::AbstractVector{<:Complex},
                               op::MHDStabilityOperator;
                               Nθ::Union{Int,Nothing}=nothing,
                               Nr::Union{Int,Nothing}=nothing,
                               interior_dofs=nothing,
                               grid::Union{MeridionalGrid,Nothing}=nothing)
    isempty(op.ll_f) && error(
        "perturbation_magnetic: this MHD problem has no magnetic field " *
        "(B0_type=no_field). Nothing to reconstruct.")
    full     = _mhd_full_vector(evec, op, interior_dofs)
    idx_map  = _mhd_index_map(op)
    g = grid === nothing ?
        build_meridional_grid(Nθ === nothing ? 2 * op.params.lmax : Nθ,
                              op.params.m, op.params.lmax) : grid
    r_grid = _mhd_radial_grid(op; Nr = Nr === nothing ? op.params.N + 1 : Nr)
    Fr, Fθ, Fφ = _mhd_poltor_to_physical(full, idx_map, op,
                                          op.ll_f, :f, op.ll_g, :g, r_grid, g)
    return Fr, Fθ, Fφ, r_grid, g
end
