# =============================================================================
#  MHD Operator Functions
#
#  Individual operator functions for MHD stability analysis:
#  - Lorentz forces (magnetic → velocity coupling)
#  - Induction operators (velocity → magnetic coupling)
#  - Magnetic diffusion
#
#  Following Kore's operators.py implementation
# =============================================================================

"""
MHD operator function implementations.
Must be included after MHD/types.jl.
"""

# This file is included via MHD/MHD.jl after types.jl
# All functions are in the Cross module scope

# -----------------------------------------------------------------------------
# Bessel Function Utilities for Conducting Inner Core BCs
# -----------------------------------------------------------------------------

@inline _complex_eltype(::Type{T}) where {T<:Real} = Complex{T}
@inline _complex_eltype(::Type{Complex{T}}) where {T<:Real} = Complex{T}

@inline function _typed_sparse(::Type{T}, mat) where {T}
    return SparseMatrixCSC{T, Int}(mat)
end

@inline function _scale_sparse(::Type{T}, coef, mat) where {T<:Real}
    Tout = (eltype(mat) <: Complex || !(coef isa Real)) ? Complex{T} : T
    return Tout(coef) * _typed_sparse(Tout, mat)
end

"""Return a correctly sized sparse complex zero block for one radial mode."""
@inline function zero_block(op::MHDStabilityOperator{T}) where {T}
    n = op.params.N + 1
    return spzeros(Complex{T}, n, n)
end

"""Return a correctly sized sparse real zero block for one radial mode."""
@inline function real_zero_block(op::MHDStabilityOperator{T}) where {T}
    n = op.params.N + 1
    return spzeros(T, n, n)
end

"""Combine `(coefficient, sparse_matrix)` terms into one sparse radial block."""
function combine_terms(terms::AbstractVector{<:Tuple})
    isempty(terms) && return spzeros(_empty_terms_output_eltype(eltype(terms)), 0, 0)
    Tmat = eltype(terms[1][2])
    Tout = (Tmat <: Complex || any(term -> !(term[1] isa Real), terms)) ?
           _complex_eltype(Tmat) : Tmat
    return _combine_terms(Tout, terms)
end

@inline _empty_terms_output_eltype(::Type) = Float64
@inline function _empty_terms_output_eltype(
    ::Type{<:Tuple{C, <:SparseMatrixCSC{T, Int}}}) where {C, T}
    return (T <: Complex || !(C <: Real)) ? _complex_eltype(T) : T
end

"""Shared implementation for `combine_terms` with a chosen output scalar type."""
function _combine_terms(::Type{T}, terms) where {T}
    isempty(terms) && return spzeros(T, 0, 0)
    rows, cols = size(terms[1][2])
    out = spzeros(T, rows, cols)
    for (coef, mat) in terms
        c = T(coef)
        c == zero(T) && continue
        out = out + c * _typed_sparse(T, mat)
    end
    return out
end

# Note: spherical_bessel_j_logderiv is now defined in boundary_conditions.jl

# -----------------------------------------------------------------------------
# Axial-field Lorentz helper functions (match Kore exactly)
# -----------------------------------------------------------------------------

"""Axial-field Lorentz coupling from poloidal magnetic field into poloidal velocity."""
function lorentz_upol_bpol_axial(op::MHDStabilityOperator{T},
                                 l::Int, m::Int, offset::Int,
                                 Le::T) where {T}
    nblock = zero_block(op)
    mat(p, h, d) = Matrix{Complex{T}}(background_operator(op, p, h, d))
    L = l * (l + 1)
    Le2 = Le^2

    if offset == -2
        denom = 3 - 8l + 4l^2
        if abs(denom) < eps(T)
            return nblock
        end
        sqrt_factor = sqrt(max((l - m) * (-1 + l + m) * (-1 + l - m) * (l + m), 0))
        if sqrt_factor == 0.0
            return nblock
        end
        C = (3 * (-2 - l + l^2) * sqrt_factor) / denom
        out = (2l + 3l^2 + l^3) .* mat(0, 0, 0)
        out .-= (6 - 7l + 3l^2) .* mat(1, 0, 1)
        out .+= (2 + l - 6l^2 + l^3) .* mat(1, 1, 0)
        out .+= (6 - l) .* mat(2, 0, 2)
        out .+= 2 * (2 - l) .* mat(2, 1, 1)
        out .+= (-2 + l) .* mat(2, 2, 0)
        out .-= mat(3, 2, 1)
        out .+= 3 .* mat(3, 0, 3)
        out .+= (3 - l) .* mat(3, 1, 2)
        out .+= (-1 + l) .* mat(3, 3, 0)
        return _scale_sparse(T, Le2 * C, sparse(out))
    elseif offset == -1
        denom = 2l - 1
        if abs(denom) < eps(T)
            return nblock
        end
        sqrt_factor = sqrt(max(l^2 - m^2, 0))
        if sqrt_factor == 0.0
            return nblock
        end
        C = sqrt_factor * (l^2 - 1) / denom
        out = -2 * (l^2 + 2) .* mat(1, 0, 1)
        out .-= 2 * (l - 2) .* mat(2, 1, 1)
        out .-= (l - 4) .* mat(2, 0, 2)
        out .-= (l - 2) .* mat(3, 1, 2)
        out .+= L * (l + 2) .* mat(0, 0, 0)
        out .+= L * (l - 4) .* mat(1, 1, 0)
        out .+= l .* mat(2, 2, 0)
        out .+= l .* mat(3, 3, 0)
        out .+= 2 .* mat(3, 0, 3)
        return _scale_sparse(T, Le2 * C, sparse(out))
    elseif offset == 0
        denom = -3 + 4l * (1 + l)
        if abs(denom) < eps(T)
            return nblock
        end
        C = 3 * (l + l^2 - 3 * m^2) / denom
        out = 3 * l * (1 + l) * (-2 + l + l^2) .* mat(0, 0, 0)
        out .-= 3 * L^2 .* mat(1, 1, 0)
        out .+= 2 * (6 - 4l - 5l^2 - 2l^3 - l^4) .* mat(1, 0, 1)
        out .+= 3 * L .* mat(2, 2, 0)
        out .+= (-12 + 5l + 5l^2) .* mat(2, 0, 2)
        out .+= 2 * (-6 + 5l + 5l^2) .* mat(2, 1, 1)
        out .+= 2 * L .* mat(3, 2, 1)
        out .+= L .* mat(3, 3, 0)
        out .+= 2 * (-3 + l + l^2) .* mat(3, 0, 3)
        out .+= 3 * (-2 + l + l^2) .* mat(3, 1, 2)
        return _scale_sparse(T, Le2 * C, sparse(out))
    elseif offset == 1
        denom = 2l + 3
        if abs(denom) < eps(T)
            return nblock
        end
        sqrt_factor = sqrt(max((l + 1 + m) * (l + 1 - m), 0))
        if sqrt_factor == 0.0
            return nblock
        end
        C = sqrt_factor * l * (l + 2) / denom
        out = -2 * (l^2 + 2l + 3) .* mat(1, 0, 1)
        out .+= 2 * (l + 3) .* mat(2, 1, 1)
        out .+= (l + 5) .* mat(2, 0, 2)
        out .+= (l + 3) .* mat(3, 1, 2)
        out .-= L * (l - 1) .* mat(0, 0, 0)
        out .-= L * (l + 5) .* mat(1, 1, 0)
        out .-= (l + 1) .* mat(2, 2, 0)
        out .-= (l + 1) .* mat(3, 3, 0)
        out .+= 2 .* mat(3, 0, 3)
        return _scale_sparse(T, Le2 * C, sparse(out))
    elseif offset == 2
        denom = (3 + 2l) * (5 + 2l)
        if abs(denom) < eps(T)
            return nblock
        end
        sqrt_factor = sqrt(max((1 + l - m) * (2 + l + m) * (1 + l + m) * (2 + l - m), 0))
        if sqrt_factor == 0.0
            return nblock
        end
        C = (3 * l * (l + 3) * sqrt_factor) / denom
        out = (l - l^3) .* mat(0, 0, 0)
        out .-= (16 + 13l + 3l^2) .* mat(1, 0, 1)
        out .-= (6 + 16l + 9l^2 + l^3) .* mat(1, 1, 0)
        out .+= 2 * (3 + l) .* mat(2, 1, 1)
        out .-= (3 + l) .* mat(2, 2, 0)
        out .+= (7 + l) .* mat(2, 0, 2)
        out .-= mat(3, 2, 1)
        out .+= 3 .* mat(3, 0, 3)
        out .+= (4 + l) .* mat(3, 1, 2)
        out .-= (2 + l) .* mat(3, 3, 0)
        return _scale_sparse(T, Le2 * C, sparse(out))
    else
        return nblock
    end
end

"""Axial-field Lorentz coupling from toroidal magnetic field into poloidal velocity."""
function lorentz_upol_btor_axial(op::MHDStabilityOperator{T},
                                 l::Int, m::Int, offset::Int,
                                 Le::T) where {T}
    nblock = zero_block(op)
    mat(p, h, d) = Matrix{Complex{T}}(background_operator(op, p, h, d))
    Le2 = Le^2

    if offset == -1
        denom = 2l - 1
        if abs(denom) < eps(T)
            return nblock
        end
        sqrt_factor = sqrt(max(l^2 - m^2, 0))
        if sqrt_factor == 0.0
            return nblock
        end
        C = (6im * m * sqrt_factor) / denom
        out = -(3 - 3l - 2l^2) .* mat(1, 0, 0)
        out .-= (l - 3) .* mat(2, 0, 1)
        out .+= (3 - 2l - l^2) .* mat(2, 1, 0)
        out .+= 3 .* mat(3, 0, 2)
        out .-= (l - 3) .* mat(3, 1, 1)
        out .-= l .* mat(3, 2, 0)
        return _scale_sparse(T, Le2 * C, sparse(out))
    elseif offset == 0
        out = -mat(1, 0, 0)
        out .-= (l^2 + l - 1) .* mat(2, 1, 0)
        out .+= mat(2, 0, 1)
        out .+= mat(3, 1, 1)
        out .+= mat(3, 0, 2)
        return _scale_sparse(T, Le2 * (2im * m), sparse(out))
    elseif offset == 1
        denom = 2l + 3
        if abs(denom) < eps(T)
            return nblock
        end
        sqrt_factor = sqrt(max((l + 1)^2 - m^2, 0))
        if sqrt_factor == 0.0
            return nblock
        end
        C = (6im * m * sqrt_factor) / denom
        out = (-4 + l + 2l^2) .* mat(1, 0, 0)
        out .+= (4 + l) .* mat(2, 0, 1)
        out .+= (4 - l^2) .* mat(2, 1, 0)
        out .+= 3 .* mat(3, 0, 2)
        out .+= (1 + l) .* mat(3, 2, 0)
        out .+= (4 + l) .* mat(3, 1, 1)
        return _scale_sparse(T, Le2 * C, sparse(out))
    else
        return nblock
    end
end

# -----------------------------------------------------------------------------
# Lorentz Force Operators (magnetic field → velocity)
# -----------------------------------------------------------------------------

"""
    operator_lorentz_poloidal(op, l, m, Le)

Lorentz force acting on poloidal velocity from background magnetic field.
Implements Kore's operators.py Lorentz force terms.

For axial background field B₀ = B₀ẑ:
- Couples toroidal magnetic perturbation g to poloidal velocity u
- Strength controlled by Lehnert number Le

Returns operator for diagonal (l) and off-diagonal (l±1) couplings.
"""
function operator_lorentz_poloidal_diagonal(op::MHDStabilityOperator{T},
                                            l::Int, Le::T) where {T}
    params = op.params
    if params.B0_type == axial
        return lorentz_upol_btor_axial(op, l, params.m, 0, Le)
    end

    m = params.m
    is_dipole = is_dipole_case(params.B0_type, params.ricb)
    shift = radial_power_shift_poloidal(is_dipole)
    bo(p, h, d) = background_operator(op, p + shift, h, d)

    terms = [
        (-1.0, bo(1, 0, 0)),
        (-(l^2 + l - 1), bo(2, 1, 0)),
        (1.0, bo(2, 0, 1)),
        (1.0, bo(3, 1, 1)),
        (1.0, bo(3, 0, 2)),
    ]

    combo = combine_terms(terms)
    return _scale_sparse(T, (Le^2) * (2im * m), combo)
end

"""Return off-diagonal toroidal-magnetic to poloidal-velocity Lorentz coupling."""
function operator_lorentz_poloidal_offdiag(op::MHDStabilityOperator{T},
                                           l::Int, m::Int, offset::Int,
                                           Le::T) where {T}
    params = op.params
    if params.B0_type == axial
        return lorentz_upol_btor_axial(op, l, m, offset, Le)
    end

    is_dipole = is_dipole_case(op.params.B0_type, op.params.ricb)
    shift = radial_power_shift_poloidal(is_dipole)
    bo(p, h, d) = background_operator(op, p + shift, h, d)

    if offset == -1
        denom = 2l - 1
        abs(denom) < eps(T) && return zero_block(op)
        coef = (Le^2) * (6im * m * sqrt(max(l^2 - m^2, 0))) / denom
        terms = [
            (-(3 - 3l - 2l^2), bo(1, 0, 0)),
            (-(l - 3), bo(2, 0, 1)),
            ((3 - 2l - l^2), bo(2, 1, 0)),
            (3.0, bo(3, 0, 2)),
            (-(l - 3), bo(3, 1, 1)),
            (-l, bo(3, 2, 0)),
        ]
        combo = combine_terms(terms)
        return _scale_sparse(T, coef, combo)
    elseif offset == 1
        denom = 2l + 3
        abs(denom) < eps(T) && return zero_block(op)
        coef = (Le^2) * (6im * m * sqrt(max((l + 1)^2 - m^2, 0))) / denom
        terms = [
            ((-4 + l + 2l^2), bo(1, 0, 0)),
            ((4 + l), bo(2, 0, 1)),
            ((4 - l^2), bo(2, 1, 0)),
            (3.0, bo(3, 0, 2)),
            ((1 + l), bo(3, 2, 0)),
            ((4 + l), bo(3, 1, 1)),
        ]
        combo = combine_terms(terms)
        return _scale_sparse(T, coef, combo)
    else
        return zero_block(op)
    end
end

"""Return poloidal-magnetic to poloidal-velocity Lorentz coupling for offsets -2:2."""
function operator_lorentz_poloidal_from_bpol(op::MHDStabilityOperator{T},
                                             l::Int, m::Int, offset::Int,
                                             Le::T) where {T}
    params = op.params
    if params.B0_type == axial
        return lorentz_upol_bpol_axial(op, l, m, offset, Le)
    end

    is_dipole = is_dipole_case(op.params.B0_type, op.params.ricb)
    shift = radial_power_shift_poloidal(is_dipole)
    bo(p, h, d) = background_operator(op, p + shift, h, d)
    L = l * (l + 1)
    Np1 = op.params.N + 1
    zero_block = spzeros(T, Np1, Np1)

    if offset == -2
        denom = 3 - 8l + 4l^2
        abs(denom) < eps() && return zero_block
        sqrt_factor = sqrt(max((l - m) * (-1 + l + m) * (-1 + l - m) * (l + m), 0))
        coef = (Le^2) * (3 * (-2 - l + l^2) * sqrt_factor) / denom
        terms = [
            (2l + 3l^2 + l^3, bo(0, 0, 0)),
            ((-6 + 7l - 3l^2), bo(1, 0, 1)),
            (2 + l - 6l^2 + l^3, bo(1, 1, 0)),
            ((6 - l), bo(2, 0, 2)),
            (2 * (2 - l), bo(2, 1, 1)),
            ((-2 + l), bo(2, 2, 0)),
            (-1.0, bo(3, 2, 1)),
            (3.0, bo(3, 0, 3)),
            ((3 - l), bo(3, 1, 2)),
            ((-1 + l), bo(3, 3, 0)),
        ]
        combo = combine_terms(terms)
        return _scale_sparse(T, coef, combo)
    elseif offset == -1
        denom = 2l - 1
        abs(denom) < eps() && return zero_block
        coef = (Le^2) * (sqrt(max(l^2 - m^2, 0)) * (l^2 - 1)) / denom
        terms = [
            (L * (l + 2), bo(0, 0, 0)),
            (L * (l - 4), bo(1, 1, 0)),
            (l, bo(2, 2, 0)),
            (l, bo(3, 3, 0)),
            (2.0, bo(3, 0, 3)),
            (-2 * (l^2 + 2), bo(1, 0, 1)),
            (-2 * (l - 2), bo(2, 1, 1)),
            ((-l + 4), bo(2, 0, 2)),
            ((-l + 2), bo(3, 1, 2)),
        ]
        combo = combine_terms(terms)
        return _scale_sparse(T, coef, combo)
    elseif offset == 0
        denom = -3 + 4l * (1 + l)
        abs(denom) < eps() && return zero_block
        coef = (Le^2) * (3 * (l + l^2 - 3m^2)) / denom
        terms = [
            (3 * l * (1 + l) * (-2 + l + l^2), bo(0, 0, 0)),
            (-3 * L^2, bo(1, 1, 0)),
            (2 * (6 - 4l - 5l^2 - 2l^3 - l^4), bo(1, 0, 1)),
            (3 * L, bo(2, 2, 0)),
            ((-12 + 5l + 5l^2), bo(2, 0, 2)),
            (2 * (-6 + 5l + 5l^2), bo(2, 1, 1)),
            (2 * L, bo(3, 2, 1)),
            (L, bo(3, 3, 0)),
            (2 * (-3 + l + l^2), bo(3, 0, 3)),
            (3 * (-2 + l + l^2), bo(3, 1, 2)),
        ]
        combo = combine_terms(terms)
        return _scale_sparse(T, coef, combo)
    elseif offset == 1
        denom = 2l + 3
        abs(denom) < eps() && return zero_block
        coef = (Le^2) * (sqrt(max((l + 1 + m) * (l + 1 - m), 0)) * l * (l + 2)) / denom
        terms = [
            (-2 * (l^2 + 2l + 3), bo(1, 0, 1)),
            (2 * (l + 3), bo(2, 1, 1)),
            ((l + 5), bo(2, 0, 2)),
            (l + 3, bo(3, 1, 2)),
            (-L * (l - 1), bo(0, 0, 0)),
            (-L * (l + 5), bo(1, 1, 0)),
            (-(l + 1), bo(2, 2, 0)),
            (-(l + 1), bo(3, 3, 0)),
            (2.0, bo(3, 0, 3)),
        ]
        combo = combine_terms(terms)
        return _scale_sparse(T, coef, combo)
    elseif offset == 2
        denom = (3 + 2l) * (5 + 2l)
        abs(denom) < eps() && return zero_block
        sqrt1 = sqrt(max((1 + l - m) * (2 + l + m), 0))
        sqrt2 = sqrt(max((2 + l - m) * (1 + l + m), 0))  # same as sqrt1 but retain symmetry
        coef = (Le^2) * (3 * l * (l + 3) * sqrt1 * sqrt2) / denom
        terms = [
            (l - l^3, bo(0, 0, 0)),
            (-(16 + 13l + 3l^2), bo(1, 0, 1)),
            (-(6 + 16l + 9l^2 + l^3), bo(1, 1, 0)),
            (2 * (3 + l), bo(2, 1, 1)),
            (-(3 + l), bo(2, 2, 0)),
            (7 + l, bo(2, 0, 2)),
            (-1.0, bo(3, 2, 1)),
            (3.0, bo(3, 0, 3)),
            (4 + l, bo(3, 1, 2)),
            (-(2 + l), bo(3, 3, 0)),
        ]
        combo = combine_terms(terms)
        return _scale_sparse(T, coef, combo)
    else
        return zero_block
    end
end

"""Return the diagonal poloidal-magnetic to toroidal-velocity Lorentz block."""
function operator_lorentz_toroidal(op::MHDStabilityOperator{T},
                                   l::Int, Le::T) where {T}
    L = l * (l + 1)
    m = op.params.m
    is_dipole = is_dipole_case(op.params.B0_type, op.params.ricb)
    shift = radial_power_shift_toroidal(is_dipole)

    term1 = background_operator(op, 0 + shift, 0, 1)
    term2 = background_operator(op, 0 + shift, 1, 0)
    term3 = background_operator(op, 1 + shift, 2, 0)
    term4 = background_operator(op, 1 + shift, 0, 2)

    combo = 4 * term1 - L * (2 * term2 + term3) + 2 * term4

    return _scale_sparse(T, (Le^2) * (1im * m), combo)
end

"""Return poloidal-magnetic to toroidal-velocity Lorentz coupling for offsets -1:1."""
function operator_lorentz_toroidal_from_bpol(op::MHDStabilityOperator{T},
                                             l::Int, m::Int, offset::Int,
                                             Le::T) where {T}
    Np1 = op.params.N + 1
    zero_block = spzeros(Complex{T}, Np1, Np1)
    is_dipole = is_dipole_case(op.params.B0_type, op.params.ricb)
    shift = radial_power_shift_toroidal(is_dipole)
    bo(p, h, d) = background_operator(op, p + shift, h, d)

    if offset == 0
        return operator_lorentz_toroidal(op, l, Le)
    elseif offset == -1
        denom = 2l - 1
        abs(denom) < eps() && return zero_block
        coef = (Le^2) * (3im * m * sqrt(max(l^2 - m^2, 0))) / denom
        combo = combine_terms([
            (12.0, bo(0, 0, 1)),
            (-2 * (l - 1) * l, bo(0, 1, 0)),
            (6.0, bo(1, 0, 2)),
            (-(l - 1) * l, bo(1, 2, 0)),
        ])
        return _scale_sparse(T, coef, combo)
    elseif offset == 1
        denom = 2l + 3
        abs(denom) < eps() && return zero_block
        coef = (Le^2) * (3im * m * sqrt(max((l + 1)^2 - m^2, 0))) / denom
        combo = combine_terms([
            (12.0, bo(0, 0, 1)),
            (-2 * (l + 1) * (l + 2), bo(0, 1, 0)),
            (6.0, bo(1, 0, 2)),
            (-(l + 1) * (l + 2), bo(1, 2, 0)),
        ])
        return _scale_sparse(T, coef, combo)
    else
        return zero_block
    end
end

"""Return toroidal-magnetic to toroidal-velocity Lorentz coupling for offsets -2:2."""
function operator_lorentz_toroidal_from_btor(op::MHDStabilityOperator{T},
                                             l::Int, m::Int, offset::Int,
                                             Le::T) where {T}
    Np1 = op.params.N + 1
    zero_block = spzeros(T, Np1, Np1)
    is_dipole = is_dipole_case(op.params.B0_type, op.params.ricb)
    shift = radial_power_shift_toroidal(is_dipole)
    bo(p, h, d) = background_operator(op, p + shift, h, d)

    if offset == -2
        denom = 3 - 8l + 4l^2
        abs(denom) < eps() && return zero_block
        sqrt_factor = sqrt(max((l - m) * (-1 + l + m) * (-1 + l - m) * (l + m), 0))
        coef = (Le^2) * (3 * (l - 2) * (l + 1) * sqrt_factor) / denom
        combo = combine_terms([
            ((-4 + l), bo(0, 0, 0)),
            (-3.0, bo(1, 0, 1)),
            ((-1 + l), bo(1, 1, 0)),
        ])
        return _scale_sparse(T, coef, combo)
    elseif offset == -1
        denom = 2l - 1
        abs(denom) < eps() && return zero_block
        coef = (Le^2) * (sqrt(max(l^2 - m^2, 0)) * (l^2 - 1)) / denom
        combo = combine_terms([
            ((l - 2), bo(0, 0, 0)),
            (l, bo(1, 1, 0)),
            (-2.0, bo(1, 0, 1)),
        ])
        return _scale_sparse(T, coef, combo)
    elseif offset == 0
        denom = -3 + 4l * (l + 1)
        abs(denom) < eps() && return zero_block
        coef = (Le^2) * (3 * (l + l^2 - 3 * m^2)) / denom
        combo = combine_terms([
            ((6 - l - l^2), bo(0, 0, 0)),
            (l * (l + 1), bo(1, 1, 0)),
            (-2 * (-3 + l + l^2), bo(1, 0, 1)),
        ])
        return _scale_sparse(T, coef, combo)
    elseif offset == 1
        denom = 2l + 3
        abs(denom) < eps() && return zero_block
        coef = (Le^2) * (-sqrt(max((l + 1 - m) * (l + 1 + m), 0)) * l * (l + 2)) / denom
        combo = combine_terms([
            ((l + 3), bo(0, 0, 0)),
            ((l + 1), bo(1, 1, 0)),
            (2.0, bo(1, 0, 1)),
        ])
        return _scale_sparse(T, coef, combo)
    elseif offset == 2
        denom = (3 + 2l) * (5 + 2l)
        abs(denom) < eps() && return zero_block
        sqrt1 = sqrt(max((2 + l - m) * (1 + l + m), 0))
        sqrt2 = sqrt(max((1 + l - m) * (2 + l + m), 0))
        coef = (Le^2) * (3 * l * (l + 3) * sqrt1 * sqrt2) / denom
        combo = combine_terms([
            (-(5 + l), bo(0, 0, 0)),
            (-3.0, bo(1, 0, 1)),
            (-(2 + l), bo(1, 1, 0)),
        ])
        return _scale_sparse(T, coef, combo)
    else
        return zero_block
    end
end

# -----------------------------------------------------------------------------
# Induction Equation Operators (velocity → magnetic field)
# -----------------------------------------------------------------------------

"""
    operator_induction_poloidal(op, l, m)

Induction of poloidal magnetic field by velocity advection.
Implements Kore's induction equation for section f.

∂B/∂t = ∇×(u × B₀) - ∇×(η∇×B)

For poloidal field (no-curl equation):
- Advection by poloidal velocity u
- Advection by toroidal velocity v
- Shear of background field
"""
function operator_induction_poloidal_from_u(op::MHDStabilityOperator{T},
                                           l::Int, m::Int, offset::Int) where {T}
    is_dipole = is_dipole_case(op.params.B0_type, op.params.ricb)
    shift = radial_power_shift_magnetic_poloidal(is_dipole)
    bo(p, h, d) = background_operator(op, p + shift, h, d)

    if offset == -2
        denom = 3 - 8l + 4l^2
        abs(denom) < eps(T) && return real_zero_block(op)
        sqrt_factor = sqrt(max((l - m) * (-1 + l + m) * (-1 + l - m) * (l + m), 0))
        C = 3 * (l - 2) * (l + 1) * sqrt_factor / denom
        terms = [
            (l - 4, bo(0, 0, 0)),
            (-3.0, bo(1, 0, 1)),
            (l - 1, bo(1, 1, 0))
        ]
        return _scale_sparse(T, C, combine_terms(terms))

    elseif offset == -1
        denom = 2l - 1
        abs(denom) < eps(T) && return real_zero_block(op)
        C = sqrt(max(l^2 - m^2, 0)) * (l^2 - 1) / denom
        terms = [
            ((l - 2), bo(0, 0, 0)),
            (l, bo(1, 1, 0)),
            (-2.0, bo(1, 0, 1))
        ]
        return _scale_sparse(T, C, combine_terms(terms))

    elseif offset == 0
        denom = -3 + 4l * (1 + l)
        abs(denom) < eps(T) && return real_zero_block(op)
        C = 3 * (l + l^2 - 3m^2) / denom
        terms = [
            ((6 - l - l^2), bo(0, 0, 0)),
            (l * (1 + l), bo(1, 1, 0)),
            (2 * (3 - l - l^2), bo(1, 0, 1))
        ]
        return _scale_sparse(T, C, combine_terms(terms))

    elseif offset == 1
        denom = 2l + 3
        abs(denom) < eps(T) && return real_zero_block(op)
        C = sqrt(max((l + 1)^2 - m^2, 0)) * l * (l + 2) / denom
        terms = [
            (-(l + 3), bo(0, 0, 0)),
            (-(l + 1), bo(1, 1, 0)),
            (-2.0, bo(1, 0, 1))
        ]
        return _scale_sparse(T, C, combine_terms(terms))

    elseif offset == 2
        denom = (3 + 2l) * (5 + 2l)
        abs(denom) < eps(T) && return real_zero_block(op)
        sqrt1 = sqrt(max((2 + l - m) * (1 + l + m), 0))
        sqrt2 = sqrt(max((1 + l - m) * (2 + l + m), 0))
        C = 3 * l * (l + 3) * sqrt1 * sqrt2 / denom
        terms = [
            (-(l + 5), bo(0, 0, 0)),
            (-3.0, bo(1, 0, 1)),
            (-(l + 2), bo(1, 1, 0))
        ]
        return _scale_sparse(T, C, combine_terms(terms))
    else
        error("offset must be in -2:-1:2 for induction poloidal (from u)")
    end
end

"""Return toroidal-velocity to poloidal-magnetic induction coupling."""
function operator_induction_poloidal_from_v(op::MHDStabilityOperator{T},
                                            l::Int, m::Int, offset::Int) where {T}
    is_dipole = is_dipole_case(op.params.B0_type, op.params.ricb)
    shift = radial_power_shift_magnetic_poloidal(is_dipole)
    term = SparseMatrixCSC{Complex{T}, Int}(background_operator(op, 1 + shift, 0, 0))
    zero_block = spzeros(Complex{T}, op.params.N + 1, op.params.N + 1)

    if offset == -1
        denom = 1 - 2l
        abs(denom) < eps(T) && return zero_block
        coef = 18im * m * sqrt(max(l^2 - m^2, 0)) / denom
        return _scale_sparse(T, coef, term)
    elseif offset == 0
        return _scale_sparse(T, -2im * m, term)
    elseif offset == 1
        denom = 3 + 2l
        abs(denom) < eps(T) && return zero_block
        coef = -18im * m * sqrt(max((l + 1)^2 - m^2, 0)) / denom
        return _scale_sparse(T, coef, term)
    else
        return zero_block
    end
end

"""
    operator_induction_toroidal(op, l, m)

Induction of toroidal magnetic field by velocity advection.
Implements Kore's induction equation for section g.
"""
function operator_induction_toroidal_from_u(op::MHDStabilityOperator{T},
                                           l::Int, m::Int, offset::Int) where {T}
    is_dipole = is_dipole_case(op.params.B0_type, op.params.ricb)
    shift = radial_power_shift_magnetic_toroidal(is_dipole)

    if offset == -1
        denom = 2l - 1
        denom == 0 && return zero_block(op)
        coef = (3im * m * sqrt(max(l^2 - m^2, 0))) / denom

        term1 = background_operator(op, 0 + shift, 1, 0)
        term2 = background_operator(op, 0 + shift, 0, 1)
        term3 = background_operator(op, -1 + shift, 0, 0)
        term4 = background_operator(op, 1 + shift, 0, 2)
        term5 = background_operator(op, 1 + shift, 1, 1)
        term6 = background_operator(op, 1 + shift, 2, 0)

        combo = -2 * ( -3 + l) * term1
        combo += -2 * (-3 + l) * term2
        combo += -2 * (3 + l^2) * term3
        combo += 6 * term4
        combo += -2 * (-3 + l) * term5
        combo += ( -1 + l) * l * term6

        return _scale_sparse(T, coef, combo)

    elseif offset == 0
        L = l * (l + 1)
        term1 = background_operator(op, 0 + shift, 0, 1)
        term2 = background_operator(op, 1 + shift, 1, 1)
        term3 = background_operator(op, -1 + shift, 0, 0)
        term4 = background_operator(op, 0 + shift, 1, 0)
        term5 = background_operator(op, 1 + shift, 2, 0)
        term6 = background_operator(op, 1 + shift, 0, 2)

        combo = term1 + term2 - (L + 1) * term3 + term4 + (L / 2) * term5 + term6
        return _scale_sparse(T, 2im * m, combo)

    elseif offset == 1
        denom = 2l + 3
        coef = (3im * m * sqrt(max((l + 1)^2 - m^2, 0))) / denom

        term1 = background_operator(op, 0 + shift, 1, 0)
        term2 = background_operator(op, 0 + shift, 0, 1)
        term3 = background_operator(op, -1 + shift, 0, 0)
        term4 = background_operator(op, 1 + shift, 0, 2)
        term5 = background_operator(op, 1 + shift, 2, 0)
        term6 = background_operator(op, 1 + shift, 1, 1)

        combo = 2 * (4 + l) * term1
        combo += 2 * (4 + l) * term2
        combo += -2 * (4 + 2l + l^2) * term3
        combo += 6 * term4
        combo += (2 + 3l + l^2) * term5
        combo += 2 * (4 + l) * term6

        return _scale_sparse(T, coef, combo)
    else
        error("offset must be -1, 0, or 1 for induction from u")
    end
end

"""Return toroidal-velocity to toroidal-magnetic induction coupling for offsets -2:2."""
function operator_induction_toroidal_from_v(op::MHDStabilityOperator{T},
                                            l::Int, m::Int, offset::Int) where {T}
    Np1 = op.params.N + 1
    zero_block = spzeros(T, Np1, Np1)
    is_dipole = is_dipole_case(op.params.B0_type, op.params.ricb)
    shift = radial_power_shift_magnetic_toroidal(is_dipole)
    bo(p, h, d) = background_operator(op, p + shift, h, d)

    if offset == -2
        denom = 3 - 8l + 4l^2
        abs(denom) < eps() && return zero_block
        sqrt_factor = sqrt(max((l - m) * (-1 + l + m) * (-1 + l - m) * (l + m), 0))
        coef = (3 * (l - 2) * (l + 1) * sqrt_factor) / denom
        combo = combine_terms([
            (l, bo(0, 0, 0)),
            (-3.0, bo(1, 0, 1)),
            ((l - 3), bo(1, 1, 0)),
        ])
        return _scale_sparse(T, coef, combo)
    elseif offset == -1
        denom = 2l - 1
        abs(denom) < eps() && return zero_block
        coef = (sqrt(max(l^2 - m^2, 0)) * (l^2 - 1)) / denom
        combo = combine_terms([
            (l, bo(0, 0, 0)),
            ((l - 2), bo(1, 1, 0)),
            (-2.0, bo(1, 0, 1)),
        ])
        return _scale_sparse(T, coef, combo)
    elseif offset == 0
        denom = -3 + 4l * (l + 1)
        abs(denom) < eps() && return zero_block
        coef = (3 * (l + l^2 - 3 * m^2)) / denom
        combo = combine_terms([
            (-(l * (l + 1)), bo(0, 0, 0)),
            (-3 * (-2 + l + l^2), bo(1, 1, 0)),
            (-2 * (-3 + l + l^2), bo(1, 0, 1)),
        ])
        return _scale_sparse(T, coef, combo)
    elseif offset == 1
        denom = 2l + 3
        abs(denom) < eps() && return zero_block
        coef = (sqrt(max((l + 1 - m) * (l + 1 + m), 0)) * l * (l + 2)) / denom
        combo = combine_terms([
            (-(l + 1), bo(0, 0, 0)),
            (-(l + 3), bo(1, 1, 0)),
            (-2.0, bo(1, 0, 1)),
        ])
        return _scale_sparse(T, coef, combo)
    elseif offset == 2
        denom = (3 + 2l) * (5 + 2l)
        abs(denom) < eps() && return zero_block
        sqrt1 = sqrt(max((2 + l - m) * (1 + l + m), 0))
        sqrt2 = sqrt(max((1 + l - m) * (2 + l + m), 0))
        coef = (3 * l * (l + 3) * sqrt1 * sqrt2) / denom
        combo = combine_terms([
            (-(1 + l), bo(0, 0, 0)),
            (-3.0, bo(1, 0, 1)),
            (-(4 + l), bo(1, 1, 0)),
        ])
        return _scale_sparse(T, coef, combo)
    else
        return zero_block
    end
end

# -----------------------------------------------------------------------------
# Magnetic Diffusion Operators
# -----------------------------------------------------------------------------

"""
    operator_magnetic_diffusion_poloidal(op, l, Em)

Magnetic diffusion for poloidal magnetic field.
∇×(η∇×B_pol)

Where Em = η/(ΩL²) is the magnetic Ekman number.
"""
function operator_magnetic_diffusion_poloidal(op::MHDStabilityOperator{T},
                                              l::Int, Em::T) where {T}
    L = l * (l + 1)
    is_dipole = is_dipole_case(op.params.B0_type, op.params.ricb)

    # Diffusion: Em * ∇²B (no-curl equation)
    # Following Kore operators.py lines 656-670
    if is_dipole
        return Em * L * (-L * op.r2_D0_f + 2 * op.r3_D1_f + op.r4_D2_f)
    end
    return Em * L * (-L * op.r0_D0_f + 2 * op.r1_D1_f + op.r2_D2_f)
end

"""
    operator_magnetic_diffusion_toroidal(op, l, Em)

Magnetic diffusion for toroidal magnetic field.
∇×(η∇×B_tor)

More complex due to spherical geometry.
"""
function operator_magnetic_diffusion_toroidal(op::MHDStabilityOperator{T},
                                              l::Int, Em::T) where {T}
    L = l * (l + 1)
    is_dipole = is_dipole_case(op.params.B0_type, op.params.ricb)

    # Toroidal magnetic diffusion
    # Following Kore operators.py lines 675-680
    # More terms than poloidal due to curl-curl in spherical coordinates
    if is_dipole
        return Em * L * (-L * op.r3_D0_g + 2 * op.r4_D1_g + op.r5_D2_g)
    end
    return Em * L * (-L * op.r0_D0_g + 2 * op.r1_D1_g + op.r2_D2_g)
end

# -----------------------------------------------------------------------------
# Time Derivative Operators for Magnetic Fields
# -----------------------------------------------------------------------------

"""
    operator_b_poloidal(op, l)

Time derivative operator for poloidal magnetic field (B matrix).
Implements op.b(l, 'b', 'bpol', 0) from Kore.

For poloidal field: r²D⁰
"""
function operator_b_poloidal(op::MHDStabilityOperator{T}, l::Int) where {T}
    # Time derivative: ∂B_pol/∂t
    # Weighted by r² for no-curl equation
    L = l * (l + 1)
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return L * op.r4_D0_f
    end
    return L * op.r2_D0_f
end

"""
    operator_b_toroidal(op, l)

Time derivative operator for toroidal magnetic field (B matrix).
Implements op.b(l, 'b', 'btor', 0) from Kore.

For toroidal field: r²D⁰
"""
function operator_b_toroidal(op::MHDStabilityOperator{T}, l::Int) where {T}
    # Time derivative: ∂B_tor/∂t
    # Weighted by r² for 1curl equation
    L = l * (l + 1)
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return L * op.r5_D0_g
    end
    return L * op.r2_D0_g
end

# Note: apply_magnetic_boundary_conditions! is now defined in boundary_conditions.jl
