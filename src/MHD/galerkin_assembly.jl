# =============================================================================
#  Tau-free ultraspherical-Galerkin assembly of the MHD HYDRO sector
#  (poloidal velocity u, toroidal velocity v, temperature h).
#
#  Builds A x = λ B x with boundary conditions carried by a recombined trial
#  basis (no tau rows → full-rank B → no spurious eigenvalues), instead of the
#  tau row-replacement used by `assemble_mhd_matrices`. Validated to reproduce
#  the collocation onset spectrum to ~1e-12 with zero spurious for the
#  MHD(B0→0) reduction. Magnetic sector (f, g) is added separately.
#
#  This is ADDITIVE: it does not modify the tau path.
# =============================================================================

"""
    assemble_mhd_galerkin(op) -> (A, B, layout)

Assemble the hydro-sector MHD generalized eigenproblem in tau-free
ultraspherical-Galerkin form. Returns dense `A`, `B` (reduced; size Σ Mᵢ with
Mᵤ = N−3 for poloidal velocity, N−1 for toroidal velocity and temperature) and a
`layout` NamedTuple `(index_map, M, R, fields, nred)` for eigenvector
reconstruction. Boundary conditions follow `op.params`: mechanical `bci/bco`
(1 ⇒ no-slip, else stress-free) and thermal `bci_thermal/bco_thermal`
(0 ⇒ fixed temperature, else fixed flux). Operator content matches the audited
onset hydro operator.
"""
function assemble_mhd_galerkin(op::MHDStabilityOperator{T}) where {T}
    p = op.params
    N = p.N; ri = p.ricb; ro = one(T); gap = ro - ri; m = p.m
    E = p.E; Pr = p.Pr; Ra = p.Ra

    noslip = (p.bci == 1 && p.bco == 1)
    fixedT = (p.bci_thermal == 0 && p.bco_thermal == 0)
    Rpol = noslip ? recomb_clamped(T, N) : recomb_poloidal_stressfree(T, N, ri, ro)
    Rtor = noslip ? recomb_dirichlet(T, N) : recomb_toroidal_stressfree(T, N, ri, ro)
    Rtem = fixedT ? recomb_dirichlet(T, N) : recomb_neumann(T, N)

    Rof = Dict(:u => Rpol, :v => Rtor, :h => Rtem)
    Mof = Dict(:u => N + 1 - 4, :v => N + 1 - 2, :h => N + 1 - 2)

    idx = Dict{Tuple{Symbol,Int},UnitRange{Int}}()
    off = 0
    for (f, ls) in ((:u, op.ll_u), (:v, op.ll_v), (:h, op.ll_h)), ℓ in ls
        idx[(f, ℓ)] = (off + 1):(off + Mof[f])
        off += Mof[f]
    end
    nred = off
    A = zeros(Complex{T}, nred, nred)
    B = zeros(Complex{T}, nred, nred)

    bt(pw, d, q) = banded_radial_term(T, pw, d, q, N, ri, ro)
    gb(band, fi, fj) = galerkin_block(band, Rof[fj], Mof[fi])

    Ra_int = Ra / gap^3
    beyonce = -Ra_int * E^2 / Pr
    thermaD = E / Pr
    Uls = op.ll_u; Vls = op.ll_v; Hls = op.ll_h

    # Poloidal velocity equation (order q=4)
    for ℓ in Uls
        L = T(ℓ * (ℓ + 1)); P = idx[(:u, ℓ)]
        B[P, P] += gb(-L * (L * bt(2,0,4) - 2 * bt(3,1,4) - bt(4,2,4)), :u, :u)
        cor = (2im * m) * (-L * bt(2,0,4) + 2 * bt(3,1,4) + bt(4,2,4))
        vis = E * L * (-L * (ℓ+2) * (ℓ-1) * bt(0,0,4) + 2 * L * bt(2,2,4) - 4 * bt(3,3,4) - bt(4,4,4))
        A[P, P] += gb(cor - vis, :u, :u)
        if (ℓ - 1) in Vls
            Cm = (ℓ^2 - 1) * sqrt(max(zero(T), T(ℓ^2 - m^2))) / (2ℓ - 1)
            A[P, idx[(:v, ℓ-1)]] += gb(2 * Cm * ((ℓ-1) * bt(3,0,4) - bt(4,1,4)), :u, :v)
        end
        if (ℓ + 1) in Vls
            Cp = ℓ * (ℓ+2) * sqrt(max(zero(T), T((ℓ+m+1) * (ℓ-m+1)))) / (2ℓ + 3)
            A[P, idx[(:v, ℓ+1)]] += gb(2 * Cp * (-(ℓ+2) * bt(3,0,4) - bt(4,1,4)), :u, :v)
        end
        if ℓ in Hls
            A[P, idx[(:h, ℓ)]] += gb(beyonce * L * bt(4,0,4), :u, :h)
        end
    end

    # Temperature equation (order q=2)
    for ℓ in Hls
        L = T(ℓ * (ℓ + 1)); H = idx[(:h, ℓ)]
        B[H, H] += gb(bt(3,0,2), :h, :h)
        if ℓ in Uls
            A[H, idx[(:u, ℓ)]] += gb((L * ri / gap) * bt(0,0,2), :h, :u)
        end
        A[H, H] += gb(thermaD * (-L * bt(1,0,2) + 2 * bt(2,1,2) + bt(3,2,2)), :h, :h)
    end

    # Toroidal velocity equation (order q=2)
    for ℓ in Vls
        L = T(ℓ * (ℓ + 1)); V = idx[(:v, ℓ)]
        B[V, V] += gb(-L * bt(2,0,2), :v, :v)
        cor = (-2im * m) * bt(2,0,2)
        vis = E * L * (-L * bt(0,0,2) + 2 * bt(1,1,2) + bt(2,2,2))
        A[V, V] += gb(cor - vis, :v, :v)
        if (ℓ - 1) in Uls
            Cm = (ℓ^2 - 1) * sqrt(max(zero(T), T(ℓ^2 - m^2))) / (2ℓ - 1)
            A[V, idx[(:u, ℓ-1)]] += gb(2 * Cm * ((ℓ-1) * bt(1,0,2) - bt(2,1,2)), :v, :u)
        end
        if (ℓ + 1) in Uls
            Cp = ℓ * (ℓ+2) * sqrt(max(zero(T), T((ℓ+m+1) * (ℓ-m+1)))) / (2ℓ + 3)
            A[V, idx[(:u, ℓ+1)]] += gb(2 * Cp * (-(ℓ+2) * bt(1,0,2) - bt(2,1,2)), :v, :u)
        end
    end

    layout = (index_map = idx, M = Mof, R = Rof, fields = (:u, :v, :h), nred = nred)
    return A, B, layout
end

"""
    reconstruct_mhd_galerkin(layout, y) -> Dict{Tuple{Symbol,Int},Vector}

Lift a reduced Galerkin eigenvector `y` back to full Chebyshev coefficients per
`(field, ℓ)` block via the recombination `R[field]`.
"""
function reconstruct_mhd_galerkin(layout, y::AbstractVector)
    out = Dict{Tuple{Symbol,Int},Vector{eltype(y)}}()
    for (key, rng) in layout.index_map
        field = key[1]
        out[key] = layout.R[field] * y[rng]
    end
    return out
end

"""Full-eigenvector layout range for `(field, ℓ)`, matching `assemble_mhd_matrices`
((N+1) coefficients per mode; sections ordered u, v, f, g, h)."""
function _mhd_full_range(op::MHDStabilityOperator, field::Symbol, ℓ::Int)
    npm = op.params.N + 1
    nbu = length(op.ll_u); nbv = length(op.ll_v)
    nbf = length(op.ll_f); nbg = length(op.ll_g)
    base = if field === :u
        (findfirst(==(ℓ), op.ll_u) - 1) * npm
    elseif field === :v
        (nbu + findfirst(==(ℓ), op.ll_v) - 1) * npm
    elseif field === :h
        (nbu + nbv + nbf + nbg + findfirst(==(ℓ), op.ll_h) - 1) * npm
    else
        error("unsupported field $field")
    end
    return (base + 1):(base + npm)
end

"""Lift a reduced hydro Galerkin eigenvector to a full `op.matrix_size` vector in
the standard MHD coefficient layout (for `StabilityResult` compatibility)."""
function reconstruct_mhd_galerkin_full(op::MHDStabilityOperator, layout, y::AbstractVector)
    full = zeros(eltype(y), op.matrix_size)
    for (key, rng) in layout.index_map
        field, ℓ = key
        full[_mhd_full_range(op, field, ℓ)] = layout.R[field] * y[rng]
    end
    return full
end
