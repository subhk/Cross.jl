# =============================================================================
#  Tau-free ultraspherical-Galerkin assembly of the MHD eigenproblem.
#
#  Boundary conditions are carried by a recombined trial basis (no tau rows →
#  full-rank B → no spurious eigenvalues). Validated to reproduce the collocation
#  onset spectrum to ~1e-12 with zero spurious for the MHD(B0→0) hydro reduction.
#
#  Sectors:
#   - Hydro (u, v, h): fully implemented (banded, B2). Matches onset to ~1e-12.
#   - Magnetic (f, g): NOT yet supported. A background field always requires Le>0
#     (MHDParams rejects Le=0 with a field), so there is no decoupled limit — the
#     velocity↔magnetic coupling (induction + Lorentz, G3.2b) is always present and
#     must be implemented together with the f/g diagonal blocks. This function
#     therefore ERRORS on any magnetic field; the caller routes such cases through
#     the tau path. The f/g diagonal mass+diffusion blocks below are scaffolding for
#     G3.2b (they are not reached until the guard is lifted and coupling is added).
#
#  ADDITIVE: does not modify the tau path (`assemble_mhd_matrices`).
# =============================================================================

"""
    assemble_mhd_galerkin(op) -> (A, B, layout)

Assemble the MHD generalized eigenproblem in tau-free ultraspherical-Galerkin
form for the hydro sector. Returns dense `A`, `B` and a `layout` NamedTuple
`(index_map, M, R, fields, nred)` for reconstruction. Errors on any magnetic
field (a field always implies `Le>0`, whose induction+Lorentz coupling is not yet
implemented — G3.2b); such cases must use the tau path.
"""
function assemble_mhd_galerkin(op::MHDStabilityOperator{T}) where {T}
    p = op.params
    N = p.N; ri = p.ricb; ro = one(T); gap = ro - ri; m = p.m
    E = p.E; Pr = p.Pr; Ra = p.Ra; Em = p.Em

    has_mag = !isempty(op.ll_f) || !isempty(op.ll_g)
    if has_mag && p.Le != 0
        error("assemble_mhd_galerkin: Le>0 magnetic coupling not implemented (G3.2b); route through the tau path.")
    end
    if has_mag && is_dipole_case(p.B0_type, p.ricb)
        error("assemble_mhd_galerkin: dipole magnetic diffusion not implemented; route through the tau path.")
    end

    noslip = (p.bci == 1 && p.bco == 1)
    fixedT = (p.bci_thermal == 0 && p.bco_thermal == 0)
    Rpol = noslip ? recomb_clamped(T, N) : recomb_poloidal_stressfree(T, N, ri, ro)
    Rtor = noslip ? recomb_dirichlet(T, N) : recomb_toroidal_stressfree(T, N, ri, ro)
    Rtem = fixedT ? recomb_dirichlet(T, N) : recomb_neumann(T, N)
    Rg   = recomb_dirichlet(T, N)                       # toroidal magnetic: g=0 both ends

    Mof = Dict(:u => N + 1 - 4, :v => N + 1 - 2, :h => N + 1 - 2,
               :f => N + 1 - 2, :g => N + 1 - 2)

    # Per-(field, ℓ) recombination (magnetic-f is ℓ-dependent Robin).
    Rmap = Dict{Tuple{Symbol,Int},Matrix{T}}()
    idx = Dict{Tuple{Symbol,Int},UnitRange{Int}}()
    off = 0
    for (f, ls) in ((:u, op.ll_u), (:v, op.ll_v), (:f, op.ll_f), (:g, op.ll_g), (:h, op.ll_h))
        for ℓ in ls
            Rmap[(f, ℓ)] = f === :u ? Matrix{T}(Rpol) :
                           f === :v ? Matrix{T}(Rtor) :
                           f === :h ? Matrix{T}(Rtem) :
                           f === :g ? Matrix{T}(Rg) :
                           recomb_magnetic_poloidal(T, N, ℓ, ri, ro; bci=p.bci_magnetic, bco=p.bco_magnetic)
            idx[(f, ℓ)] = (off + 1):(off + Mof[f])
            off += Mof[f]
        end
    end
    nred = off
    A = zeros(Complex{T}, nred, nred)
    B = zeros(Complex{T}, nred, nred)

    bt(pw, d, q) = banded_radial_term(T, pw, d, q, N, ri, ro)
    gb(band, fi, fj, ℓj) = galerkin_block(band, Rmap[(fj, ℓj)], Mof[fi])

    Ra_int = Ra / gap^3
    beyonce = -Ra_int * E^2 / Pr
    thermaD = E / Pr
    Uls = op.ll_u; Vls = op.ll_v; Hls = op.ll_h

    # ---- Poloidal velocity equation (order q=4) ----
    for ℓ in Uls
        L = T(ℓ * (ℓ + 1)); P = idx[(:u, ℓ)]
        B[P, P] += gb(-L * (L * bt(2,0,4) - 2 * bt(3,1,4) - bt(4,2,4)), :u, :u, ℓ)
        cor = (2im * m) * (-L * bt(2,0,4) + 2 * bt(3,1,4) + bt(4,2,4))
        vis = E * L * (-L * (ℓ+2) * (ℓ-1) * bt(0,0,4) + 2 * L * bt(2,2,4) - 4 * bt(3,3,4) - bt(4,4,4))
        A[P, P] += gb(cor - vis, :u, :u, ℓ)
        if (ℓ - 1) in Vls
            Cm = (ℓ^2 - 1) * sqrt(max(zero(T), T(ℓ^2 - m^2))) / (2ℓ - 1)
            A[P, idx[(:v, ℓ-1)]] += gb(2 * Cm * ((ℓ-1) * bt(3,0,4) - bt(4,1,4)), :u, :v, ℓ-1)
        end
        if (ℓ + 1) in Vls
            Cp = ℓ * (ℓ+2) * sqrt(max(zero(T), T((ℓ+m+1) * (ℓ-m+1)))) / (2ℓ + 3)
            A[P, idx[(:v, ℓ+1)]] += gb(2 * Cp * (-(ℓ+2) * bt(3,0,4) - bt(4,1,4)), :u, :v, ℓ+1)
        end
        if ℓ in Hls
            A[P, idx[(:h, ℓ)]] += gb(beyonce * L * bt(4,0,4), :u, :h, ℓ)
        end
    end

    # ---- Temperature equation (order q=2) ----
    for ℓ in Hls
        L = T(ℓ * (ℓ + 1)); H = idx[(:h, ℓ)]
        B[H, H] += gb(bt(3,0,2), :h, :h, ℓ)
        if ℓ in Uls
            A[H, idx[(:u, ℓ)]] += gb((L * ri / gap) * bt(0,0,2), :h, :u, ℓ)
        end
        A[H, H] += gb(thermaD * (-L * bt(1,0,2) + 2 * bt(2,1,2) + bt(3,2,2)), :h, :h, ℓ)
    end

    # ---- Toroidal velocity equation (order q=2) ----
    for ℓ in Vls
        L = T(ℓ * (ℓ + 1)); V = idx[(:v, ℓ)]
        B[V, V] += gb(-L * bt(2,0,2), :v, :v, ℓ)
        cor = (-2im * m) * bt(2,0,2)
        vis = E * L * (-L * bt(0,0,2) + 2 * bt(1,1,2) + bt(2,2,2))
        A[V, V] += gb(cor - vis, :v, :v, ℓ)
        if (ℓ - 1) in Uls
            Cm = (ℓ^2 - 1) * sqrt(max(zero(T), T(ℓ^2 - m^2))) / (2ℓ - 1)
            A[V, idx[(:u, ℓ-1)]] += gb(2 * Cm * ((ℓ-1) * bt(1,0,2) - bt(2,1,2)), :v, :u, ℓ-1)
        end
        if (ℓ + 1) in Uls
            Cp = ℓ * (ℓ+2) * sqrt(max(zero(T), T((ℓ+m+1) * (ℓ-m+1)))) / (2ℓ + 3)
            A[V, idx[(:u, ℓ+1)]] += gb(2 * Cp * (-(ℓ+2) * bt(1,0,2) - bt(2,1,2)), :v, :u, ℓ+1)
        end
    end

    # ---- Magnetic diagonal blocks (axial, decoupled at Le=0): mass + diffusion ----
    for ℓ in op.ll_f
        L = T(ℓ * (ℓ + 1)); F = idx[(:f, ℓ)]
        B[F, F] += gb(-L * bt(2,0,2), :f, :f, ℓ)
        A[F, F] += gb(Em * L * (-L * bt(0,0,2) + 2 * bt(1,1,2) + bt(2,2,2)), :f, :f, ℓ)
    end
    for ℓ in op.ll_g
        L = T(ℓ * (ℓ + 1)); G = idx[(:g, ℓ)]
        B[G, G] += gb(-L * bt(2,0,2), :g, :g, ℓ)
        A[G, G] += gb(Em * L * (-L * bt(0,0,2) + 2 * bt(1,1,2) + bt(2,2,2)), :g, :g, ℓ)
    end

    layout = (index_map = idx, M = Mof, R = Rmap, fields = (:u, :v, :f, :g, :h), nred = nred)
    return A, B, layout
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
    elseif field === :f
        (nbu + nbv + findfirst(==(ℓ), op.ll_f) - 1) * npm
    elseif field === :g
        (nbu + nbv + nbf + findfirst(==(ℓ), op.ll_g) - 1) * npm
    elseif field === :h
        (nbu + nbv + nbf + nbg + findfirst(==(ℓ), op.ll_h) - 1) * npm
    else
        error("unsupported field $field")
    end
    return (base + 1):(base + npm)
end

"""Lift a reduced Galerkin eigenvector to a full `op.matrix_size` vector in the
standard MHD coefficient layout (for `StabilityResult` compatibility)."""
function reconstruct_mhd_galerkin_full(op::MHDStabilityOperator, layout, y::AbstractVector)
    full = zeros(eltype(y), op.matrix_size)
    for (key, rng) in layout.index_map
        field, ℓ = key
        full[_mhd_full_range(op, field, ℓ)] = layout.R[key] * y[rng]
    end
    return full
end

"""Reduced per-block reconstruction (full Chebyshev coefficients per `(field, ℓ)`)."""
function reconstruct_mhd_galerkin(layout, y::AbstractVector)
    out = Dict{Tuple{Symbol,Int},Vector{eltype(y)}}()
    for (key, rng) in layout.index_map
        out[key] = layout.R[key] * y[rng]
    end
    return out
end
