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
Module containing MHD operator function implementations.
Must be included after MHDOperator.jl
"""

# This file is meant to be included in MHDOperator.jl
# All functions are in the MHDOperator module scope

# -----------------------------------------------------------------------------
# Bessel Function Utilities for Conducting Inner Core BCs
# -----------------------------------------------------------------------------

const SparseF64 = SparseMatrixCSC{Float64,Int}
const SparseC64 = SparseMatrixCSC{ComplexF64,Int}

@inline function complex_background_operator(op::MHDStabilityOperator,
                                             p::Int, h::Int, d::Int,
                                             shift::Int=0)
    return SparseC64(background_operator(op, p + shift, h, d))
end

@inline function zero_block(op::MHDStabilityOperator)
    n = op.params.N + 1
    return spzeros(ComplexF64, n, n)
end

function combine_terms(terms::AbstractVector{<:Tuple{T,SparseF64}}) where {T<:Real}
    isempty(terms) && return spzeros(Float64, 0, 0)
    rows, cols = size(terms[1][2])
    out = spzeros(Float64, rows, cols)
    for (coef, mat) in terms
        c = Float64(coef)
        if c != 0.0
            out += c * mat
        end
    end
    return out
end

"""
    spherical_bessel_j_logderiv(l::Int, x::Complex{T}) -> Complex{T}

Compute the logarithmic derivative of the spherical Bessel function of the first kind:

```math
\\frac{d}{dx}[\\log(j_l(x))] = \\frac{j'_l(x)}{j_l(x)}
```

# Mathematical Background

The spherical Bessel function jₗ(x) satisfies the recurrence relation:
```math
j'_l(x) = \\frac{l}{x} j_l(x) - j_{l+1}(x)
```

Therefore, the logarithmic derivative is:
```math
\\frac{j'_l(x)}{j_l(x)} = \\frac{l}{x} - \\frac{j_{l+1}(x)}{j_l(x)}
```

# Application: Conducting Inner Core

For a conducting inner core with finite conductivity σ, the magnetic boundary
condition couples the field and its derivative through a complex Bessel wavenumber:

```math
k = (1-i)\\sqrt{\\frac{\\omega}{2E_m}}
```

where ω is the oscillation frequency and Eₘ is the magnetic Ekman number.

The boundary condition becomes:
```math
f(r_i) - k \\cdot \\frac{j'_l(kr_i)}{j_l(kr_i)} \\cdot f'(r_i) = 0
```

# Numerical Stability

- For |x| < 10⁻¹⁰, uses series expansion: j'ₗ/jₗ ≈ l/x
- For normal values, uses recurrence relation
- Handles complex arguments robustly (needed for k = (1-i)√...)

# Arguments

- `l::Int`: Spherical harmonic degree (l ≥ 0)
- `x::Complex{T}`: Complex argument (typically k·r)

# Returns

- `Complex{T}`: The logarithmic derivative j'ₗ(x)/jₗ(x)

# Examples

```julia
using SpecialFunctions

# Real argument
l = 2
x = 1.5
logderiv = spherical_bessel_j_logderiv(l, x)

# Complex argument (conducting boundary)
Em = 1e-3
omega = 0.1 + 0.5im  # Complex frequency
k = (1 - 1im) * sqrt(omega / (2*Em))
ri = 0.35
logderiv_complex = spherical_bessel_j_logderiv(l, k * ri)
```

# References

- Kore implementation: kore-main/bin/utils.py, lines 487-526
- Satapathy (2013): Boundary conditions for conducting cores
- Zhang & Fearn (1994): Hydromagnetic flow in planetary cores

# See Also

- [`apply_magnetic_boundary_conditions!`](@ref): Uses this for bci_magnetic=1
- `SpecialFunctions.sphericalbesselj`: Underlying Bessel function
"""
function spherical_bessel_j_logderiv(l::Int, x::Complex{T}) where {T<:Real}
    # For very small |x|, use series expansion: j_l(x) ≈ x^l / (2l+1)!!
    # so d/dx[log(j_l)] ≈ l/x
    if abs(x) < 1e-10
        return complex(T(l)) / x
    end

    # Use recurrence relation: d/dx[log(j_l)] = l/x - j_{l+1}/j_l
    jl = sphericalbesselj(l, x)
    jl_plus_1 = sphericalbesselj(l + 1, x)

    # Check for numerical issues
    if abs(jl) < 1e-30
        # If j_l is very small, fall back to asymptotic form
        return complex(T(l)) / x
    end

    return T(l) / x - jl_plus_1 / jl
end

# Overload for real arguments (though we primarily use complex)
spherical_bessel_j_logderiv(l::Int, x::T) where {T<:Real} =
    spherical_bessel_j_logderiv(l, complex(x))

# -----------------------------------------------------------------------------
# Axial-field Lorentz helper functions (match Kore exactly)
# -----------------------------------------------------------------------------

function lorentz_upol_bpol_axial(op::MHDStabilityOperator{T},
                                 l::Int, m::Int, offset::Int,
                                 Le::T) where {T}
    nblock = zero_block(op)
    mat(p, h, d) = Matrix{ComplexF64}(background_operator(op, p, h, d))
    L = l * (l + 1)
    Le2 = Le^2

    if offset == -2
        denom = 3 - 8l + 4l^2
        if abs(denom) < eps(Float64)
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
        return sparse(Le2 .* (C .* out))
    elseif offset == -1
        denom = 2l - 1
        if abs(denom) < eps(Float64)
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
        return sparse(Le2 .* (C .* out))
    elseif offset == 0
        denom = -3 + 4l * (1 + l)
        if abs(denom) < eps(Float64)
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
        return sparse(Le2 .* (C .* out))
    elseif offset == 1
        denom = 2l + 3
        if abs(denom) < eps(Float64)
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
        return sparse(Le2 .* (C .* out))
    elseif offset == 2
        denom = (3 + 2l) * (5 + 2l)
        if abs(denom) < eps(Float64)
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
        return sparse(Le2 .* (C .* out))
    else
        return nblock
    end
end

function lorentz_upol_btor_axial(op::MHDStabilityOperator{T},
                                 l::Int, m::Int, offset::Int,
                                 Le::T) where {T}
    nblock = zero_block(op)
    mat(p, h, d) = Matrix{ComplexF64}(background_operator(op, p, h, d))
    Le2 = Le^2

    if offset == -1
        denom = 2l - 1
        if abs(denom) < eps(Float64)
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
        return sparse(Le2 .* (C .* out))
    elseif offset == 0
        out = -mat(1, 0, 0)
        out .-= (l^2 + l - 1) .* mat(2, 1, 0)
        out .+= mat(2, 0, 1)
        out .+= mat(3, 1, 1)
        out .+= mat(3, 0, 2)
        return sparse(Le2 .* (2im * m) .* out)
    elseif offset == 1
        denom = 2l + 3
        if abs(denom) < eps(Float64)
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
        return sparse(Le2 .* (C .* out))
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
    return (Le^2) * (2im * m) * combo
end

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
        abs(denom) < eps() && return spzeros(Float64, op.params.N + 1, op.params.N + 1)
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
        return coef * combo
    elseif offset == 1
        denom = 2l + 3
        abs(denom) < eps() && return spzeros(Float64, op.params.N + 1, op.params.N + 1)
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
        return coef * combo
    else
        return spzeros(Float64, op.params.N + 1, op.params.N + 1)
    end
end

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
    zero_block = spzeros(Float64, Np1, Np1)

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
        return coef * combo
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
        return coef * combo
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
        return coef * combo
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
        return coef * combo
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
        return coef * combo
    else
        return zero_block
    end
end

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

    return (Le^2) * (1im * m) * combo
end

function operator_lorentz_toroidal_from_bpol(op::MHDStabilityOperator{T},
                                             l::Int, m::Int, offset::Int,
                                             Le::T) where {T}
    Np1 = op.params.N + 1
    zero_block = spzeros(Float64, Np1, Np1)
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
        return coef * combo
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
        return coef * combo
    else
        return zero_block
    end
end

function operator_lorentz_toroidal_from_btor(op::MHDStabilityOperator{T},
                                             l::Int, m::Int, offset::Int,
                                             Le::T) where {T}
    Np1 = op.params.N + 1
    zero_block = spzeros(Float64, Np1, Np1)
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
        return coef * combo
    elseif offset == -1
        denom = 2l - 1
        abs(denom) < eps() && return zero_block
        coef = (Le^2) * (sqrt(max(l^2 - m^2, 0)) * (l^2 - 1)) / denom
        combo = combine_terms([
            ((l - 2), bo(0, 0, 0)),
            (l, bo(1, 1, 0)),
            (-2.0, bo(1, 0, 1)),
        ])
        return coef * combo
    elseif offset == 0
        denom = -3 + 4l * (l + 1)
        abs(denom) < eps() && return zero_block
        coef = (Le^2) * (3 * (l + l^2 - 3 * m^2)) / denom
        combo = combine_terms([
            ((6 - l - l^2), bo(0, 0, 0)),
            (l * (l + 1), bo(1, 1, 0)),
            (-2 * (-3 + l + l^2), bo(1, 0, 1)),
        ])
        return coef * combo
    elseif offset == 1
        denom = 2l + 3
        abs(denom) < eps() && return zero_block
        coef = (Le^2) * (-sqrt(max((l + 1 - m) * (l + 1 + m), 0)) * l * (l + 2)) / denom
        combo = combine_terms([
            ((l + 3), bo(0, 0, 0)),
            ((l + 1), bo(1, 1, 0)),
            (2.0, bo(1, 0, 1)),
        ])
        return coef * combo
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
        return coef * combo
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
        abs(denom) < eps() && return spzeros(Float64, op.params.N + 1, op.params.N + 1)
        sqrt_factor = sqrt(max((l - m) * (-1 + l + m) * (-1 + l - m) * (l + m), 0))
        C = 3 * (l - 2) * (l + 1) * sqrt_factor / denom
        terms = [
            (l - 4, bo(0, 0, 0)),
            (-3.0, bo(1, 0, 1)),
            (l - 1, bo(1, 1, 0))
        ]
        return C * combine_terms(terms)

    elseif offset == -1
        denom = 2l - 1
        abs(denom) < eps() && return spzeros(Float64, op.params.N + 1, op.params.N + 1)
        C = sqrt(max(l^2 - m^2, 0)) * (l^2 - 1) / denom
        terms = [
            ((l - 2), bo(0, 0, 0)),
            (l, bo(1, 1, 0)),
            (-2.0, bo(1, 0, 1))
        ]
        return C * combine_terms(terms)

    elseif offset == 0
        denom = -3 + 4l * (1 + l)
        abs(denom) < eps() && return spzeros(Float64, op.params.N + 1, op.params.N + 1)
        C = 3 * (l + l^2 - 3m^2) / denom
        terms = [
            ((6 - l - l^2), bo(0, 0, 0)),
            (l * (1 + l), bo(1, 1, 0)),
            (2 * (3 - l - l^2), bo(1, 0, 1))
        ]
        return C * combine_terms(terms)

    elseif offset == 1
        denom = 2l + 3
        abs(denom) < eps() && return spzeros(Float64, op.params.N + 1, op.params.N + 1)
        C = sqrt(max((l + 1)^2 - m^2, 0)) * l * (l + 2) / denom
        terms = [
            (-(l + 3), bo(0, 0, 0)),
            (-(l + 1), bo(1, 1, 0)),
            (-2.0, bo(1, 0, 1))
        ]
        return C * combine_terms(terms)

    elseif offset == 2
        denom = (3 + 2l) * (5 + 2l)
        abs(denom) < eps() && return spzeros(Float64, op.params.N + 1, op.params.N + 1)
        sqrt1 = sqrt(max((2 + l - m) * (1 + l + m), 0))
        sqrt2 = sqrt(max((1 + l - m) * (2 + l + m), 0))
        C = 3 * l * (l + 3) * sqrt1 * sqrt2 / denom
        terms = [
            (-(l + 5), bo(0, 0, 0)),
            (-3.0, bo(1, 0, 1)),
            (-(l + 2), bo(1, 1, 0))
        ]
        return C * combine_terms(terms)
    else
        error("offset must be in -2:-1:2 for induction poloidal (from u)")
    end
end

function operator_induction_poloidal_from_v(op::MHDStabilityOperator{T},
                                            l::Int, m::Int, offset::Int) where {T}
    is_dipole = is_dipole_case(op.params.B0_type, op.params.ricb)
    shift = radial_power_shift_magnetic_poloidal(is_dipole)
    term = SparseMatrixCSC{ComplexF64, Int}(background_operator(op, 1 + shift, 0, 0))
    zero_block = spzeros(ComplexF64, op.params.N + 1, op.params.N + 1)

    if offset == -1
        denom = 1 - 2l
        abs(denom) < eps() && return zero_block
        coef = 18im * m * sqrt(max(l^2 - m^2, 0)) / denom
        return coef * term
    elseif offset == 0
        return -2im * m * term
    elseif offset == 1
        denom = 3 + 2l
        abs(denom) < eps() && return zero_block
        coef = -18im * m * sqrt(max((l + 1)^2 - m^2, 0)) / denom
        return coef * term
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
        denom == 0 && return spzeros(ComplexF64, op.params.N + 1, op.params.N + 1)
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

        return coef * combo

    elseif offset == 0
        L = l * (l + 1)
        term1 = background_operator(op, 0 + shift, 0, 1)
        term2 = background_operator(op, 1 + shift, 1, 1)
        term3 = background_operator(op, -1 + shift, 0, 0)
        term4 = background_operator(op, 0 + shift, 1, 0)
        term5 = background_operator(op, 1 + shift, 2, 0)
        term6 = background_operator(op, 1 + shift, 0, 2)

        combo = term1 + term2 - (L + 1) * term3 + term4 + (L / 2) * term5 + term6
        return (2im * m) * combo

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

        return coef * combo
    else
        error("offset must be -1, 0, or 1 for induction from u")
    end
end

function operator_induction_toroidal_from_v(op::MHDStabilityOperator{T},
                                            l::Int, m::Int, offset::Int) where {T}
    Np1 = op.params.N + 1
    zero_block = spzeros(Float64, Np1, Np1)
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
        return coef * combo
    elseif offset == -1
        denom = 2l - 1
        abs(denom) < eps() && return zero_block
        coef = (sqrt(max(l^2 - m^2, 0)) * (l^2 - 1)) / denom
        combo = combine_terms([
            (l, bo(0, 0, 0)),
            ((l - 2), bo(1, 1, 0)),
            (-2.0, bo(1, 0, 1)),
        ])
        return coef * combo
    elseif offset == 0
        denom = -3 + 4l * (l + 1)
        abs(denom) < eps() && return zero_block
        coef = (3 * (l + l^2 - 3 * m^2)) / denom
        combo = combine_terms([
            (-(l * (l + 1)), bo(0, 0, 0)),
            (-3 * (-2 + l + l^2), bo(1, 1, 0)),
            (-2 * (-3 + l + l^2), bo(1, 0, 1)),
        ])
        return coef * combo
    elseif offset == 1
        denom = 2l + 3
        abs(denom) < eps() && return zero_block
        coef = (sqrt(max((l + 1 - m) * (l + 1 + m), 0)) * l * (l + 2)) / denom
        combo = combine_terms([
            (-(l + 1), bo(0, 0, 0)),
            (-(l + 3), bo(1, 1, 0)),
            (-2.0, bo(1, 0, 1)),
        ])
        return coef * combo
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
        return coef * combo
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

# -----------------------------------------------------------------------------
# Background Field Structure Functions
# -----------------------------------------------------------------------------

"""
    compute_background_field_coefficients(B0_type, N, ri, ro)

Compute Chebyshev coefficients for background field structure function h(r).

For axial field: h(r) = r
For dipole field: h(r) = 1/r² (more complex)
"""
function compute_background_field_coefficients(B0_type::BackgroundField,
                                              N::Int, ri::Float64, ro::Float64)
    if B0_type == axial
        # Axial field: h(r) = r
        # Chebyshev coefficients already computed in operator building
        return nothing  # Use r^1 operators directly
    elseif B0_type == dipole
        # Dipole field: h(r) = 1/r²
        # Need special handling for negative powers
        error("Dipole field not yet implemented - requires r^(-2) operators")
    else
        # No background field
        return nothing
    end
end

# -----------------------------------------------------------------------------
# Magnetic Boundary Conditions
# -----------------------------------------------------------------------------

"""
    apply_magnetic_boundary_conditions!(A, B, op, section)

Apply boundary conditions to magnetic field sections using the tau method.

Modifies the sparse matrices A and B in-place by replacing boundary rows with
constraint equations. This implements the tau method for enforcing boundary
conditions in spectral methods.

# Physical Boundary Conditions

## Boundary Condition Types (bci_magnetic, bco_magnetic)

- **0 = Insulating**: Electrically insulating boundary (mantle, vacuum)
  - No currents can flow across the boundary
  - Magnetic field lines can penetrate but with specific matching conditions
  - Most common for CMB (electrically insulating silicate mantle)

- **1 = Conducting**: Finite electrical conductivity (partially implemented)
  - Allows currents in a thin layer with finite conductivity σ
  - Uses Bessel functions with complex wavenumber k = (1-i)√(ω/2Eₘ)
  - **Status**: Helper functions ready, BC implementation disabled

- **2 = Perfect Conductor**: Infinite electrical conductivity
  - Tangential electric field must vanish: E_tangential = 0
  - Used for highly conducting solid inner cores (e.g., Earth's iron core)
  - Requires 2-row BC for poloidal field, 1-row for toroidal

# Mathematical Formulation

## Poloidal Magnetic Field (section = :f)

The poloidal magnetic field is represented by scalar f(r) such that:
```math
\\mathbf{B}_{pol} = \\nabla \\times (f(r) \\mathbf{r})
```

### Insulating Boundaries

Match normal component and tangential derivative:

**CMB (r = rₒ = 1):**
```math
(l+1) f(r_o) + r_o f'(r_o) = 0
```

**ICB (r = rᵢ = ricb):**
```math
l f(r_i) - r_i f'(r_i) = 0
```

*Reference*: Kore assemble.py:1494-1572; Hollerbach (2000), Phys. Earth Planet. Inter.

### Conducting ICB (Finite Conductivity)

With skin depth δ ~ √(η/ω), the boundary condition involves Bessel functions:
```math
f(r_i) - k \\cdot \\frac{j'_l(kr_i)}{j_l(kr_i)} \\cdot f'(r_i) = 0
```
where k = (1-i)√(ω/2Eₘ) is the complex magnetic diffusion wavenumber.

*Reference*: Kore assemble.py:1575-1612; Satapathy (2013)

**Status**: Implemented for `bci_magnetic=1`. For omega = 0, uses the small-k limit.

### Perfect Conductor ICB

Requires **two boundary conditions** (2-row constraint):

**Row 1**: No radial field penetration:
```math
f(r_i) = 0
```

**Row 2**: Tangential electric field vanishes:
```math
E_m \\left( -f''(r_i) - \\frac{2}{r_i}f'(r_i) + \\frac{L}{r_i^2}f(r_i) \\right) = 0
```

where L = l(l+1).

*Reference*: Kore assemble.py:1614-1630; Dormy et al. (2004), J. Fluid Mech.

### Conducting/Perfect CMB

Simply enforce no penetration:
```math
f(r_o) = 0
```

## Toroidal Magnetic Field (section = :g)

The toroidal magnetic field is:
```math
\\mathbf{B}_{tor} = \\nabla \\times \\nabla \\times (g(r) \\mathbf{r})
```

### Insulating/Conducting (Standard Cases)

No toroidal field can exist outside an insulating or finitely conducting boundary:
```math
g(r) = 0 \\quad \\text{at both boundaries}
```

### Perfect Conductor ICB

Tangential electric field condition gives:
```math
E_m \\left( -g'(r_i) - \\frac{1}{r_i}g(r_i) \\right) = 0
```

*Reference*: Kore assemble.py:1631-1641

# Implementation: Tau Method

The tau method replaces matrix rows with boundary condition equations:

1. **Zero out row**: Set A[row,:] = 0 and B[row,:] = 0
2. **Build constraint**: Evaluate boundary condition using Chebyshev basis
   - Function values: Tₙ(1) = 1 at CMB, Tₙ(-1) = (-1)ⁿ at ICB
   - Derivatives: Evaluate Chebyshev derivatives at the boundary points
3. **Set row**: A[row,:] = boundary constraint, B[row,:] = 0

# Arguments

- `A::SparseMatrixCSC`: Left-hand side matrix (modified in-place)
- `B::SparseMatrixCSC`: Right-hand side matrix (modified in-place)
- `op::MHDStabilityOperator`: Operator containing parameters and mode structure
- `section::Symbol`: Which magnetic field component (:f for poloidal, :g for toroidal)

# Matrix Structure

For each l-mode, modifies rows corresponding to boundary points:
- First row in each mode block: Outer boundary (CMB)
- Last row in each mode block: Inner boundary (ICB)
- Perfect conductor: Uses TWO rows at ICB for section :f

# Examples

```julia
# Create operator
params = MHDParams(E=1e-3, Pm=5.0, ricb=0.35, m=2, lmax=15, N=32,
                   bci_magnetic=0, bco_magnetic=0)  # Insulating
op = MHDStabilityOperator(params)

# Matrices before BC
A, B = assemble_mhd_matrices_no_bc(op)  # Hypothetical function

# Apply magnetic BCs
apply_magnetic_boundary_conditions!(A, B, op, :f)  # Poloidal field
apply_magnetic_boundary_conditions!(A, B, op, :g)  # Toroidal field

# Now A, B have correct boundary conditions enforced
```

# Boundary Condition Compatibility

| bci_magnetic | bco_magnetic | Physical Scenario | Status |
|--------------|--------------|-------------------|--------|
| 0 | 0 | Insulating mantle & vacuum outside core | ✅ Implemented |
| 0 | 1,2 | Insulating ICB, conducting CMB | ✅ Implemented (rare) |
| 2 | 0 | Perfect conductor ICB, insulating CMB | ✅ **NEW!** Earth-like |
| 1 | 0 | Conducting ICB (finite σ) | ⚠️ Infrastructure ready |

# References

- Hollerbach (2000), "A spectral solution of the magneto-convection equations...",
  Physics of the Earth and Planetary Interiors 117, 319-333
- Dormy et al. (2004), "MHD flow in a slightly differentially rotating spherical shell",
  Journal of Fluid Mechanics 501, 43-70
- Satapathy (2013), "Dynamo action in a rotating spherical shell with an outer stably stratified layer"
- Kore implementation: kore-main/bin/assemble.py:1472-1786

# See Also

- [`spherical_bessel_j_logderiv`](@ref): For conducting boundaries
- [`MHDParams`](@ref): Parameter structure with magnetic BC options
- [`assemble_mhd_matrices`](@ref): Calls this function during assembly
"""
function apply_magnetic_boundary_conditions!(A::SparseMatrixCSC,
                                            B::SparseMatrixCSC,
                                            op::MHDStabilityOperator{T},
                                            section::Symbol) where {T}
    params = op.params
    N = params.N
    n_per_mode = N + 1
    ri = params.ricb
    ro = one(T)  # Outer radius normalized to 1

    nb_u = length(op.ll_u)
    nb_v = length(op.ll_v)
    nb_f = length(op.ll_f)
    nb_g = length(op.ll_g)
    scale = UltrasphericalSpectral._radial_scale(ri, ro)
    r_outer = UltrasphericalSpectral._boundary_radius(ri, ro, :outer)
    r_inner = UltrasphericalSpectral._boundary_radius(ri, ro, :inner)
    outer_vals = UltrasphericalSpectral._chebyshev_boundary_values(N, :outer)
    inner_vals = UltrasphericalSpectral._chebyshev_boundary_values(N, :inner)
    outer_deriv = scale * UltrasphericalSpectral._chebyshev_boundary_derivative(N, :outer)
    inner_deriv = scale * UltrasphericalSpectral._chebyshev_boundary_derivative(N, :inner)
    inner_second = scale^2 * UltrasphericalSpectral._chebyshev_boundary_second_derivative(N, :inner)

    if section == :f  # Poloidal magnetic field
        for (k, l) in enumerate(op.ll_f)
            # Offset to f section: after u and v sections
            row_base = (nb_u + nb_v + k - 1) * n_per_mode
            block_range = (row_base + 1):(row_base + n_per_mode)

            # ----------------------------------------------------------------
            # Outer boundary (CMB, r = ro = 1)
            # ----------------------------------------------------------------
            row_cmb = row_base + 1

            if params.bco_magnetic == 0
                # Insulating CMB: (l+1)·f(ro) + ro·f'(ro) = 0
                # Following Kore: kore-main/bin/assemble.py:1494-1509

                # Zero out row
                A[row_cmb, :] .= zero(ComplexF64)
                B[row_cmb, :] .= zero(ComplexF64)

                # Build constraint: (l+1)·f + ro·f'
                A[row_cmb, block_range] = ComplexF64.((l + 1) * outer_vals + r_outer * outer_deriv)

            else
                # Perfectly conducting: f = 0 (no penetration)
                A[row_cmb, :] .= zero(ComplexF64)
                B[row_cmb, :] .= zero(ComplexF64)
                A[row_cmb, block_range] = ComplexF64.(outer_vals)
            end

            # ----------------------------------------------------------------
            # Inner boundary (ICB, r = ri)
            # ----------------------------------------------------------------
            row_icb = row_base + n_per_mode

            if params.bci_magnetic == 0
                # Insulating ICB: l·f(ri) - ri·f'(ri) = 0
                # Following Kore: kore-main/bin/assemble.py:1548-1572

                # Zero out row
                A[row_icb, :] .= zero(ComplexF64)
                B[row_icb, :] .= zero(ComplexF64)

                # Build constraint: l·f - ri·f'
                A[row_icb, block_range] = ComplexF64.(l * inner_vals - r_inner * inner_deriv)

            elseif params.bci_magnetic == 1
                freq = params.forcing_frequency
                Em = params.Em
                if Em <= 0
                    error("Conducting magnetic BC requires Em > 0")
                end

                # If frequency is zero (steady state), condition reduces to f(ri) = 0
                if iszero(freq)
                    A[row_icb, :] .= zero(ComplexF64)
                    B[row_icb, :] .= zero(ComplexF64)
                    A[row_icb, block_range] = ComplexF64.(r_inner .* inner_vals .- l .* inner_deriv)
                else
                    k = (1 - 1im) * sqrt(complex(freq) / (2 * Em))
                    dlog = spherical_bessel_j_logderiv(l, k * ri)

                    A[row_icb, :] .= zero(ComplexF64)
                    B[row_icb, :] .= zero(ComplexF64)
                    A[row_icb, block_range] = ComplexF64.(inner_vals) .-
                                             k * dlog .* ComplexF64.(inner_deriv)
                end

            elseif params.bci_magnetic == 2
                # Perfect conductor ICB: 2-row boundary condition
                # Row 1: f = 0
                # Row 2: Em·(-f'' - 2/ri·f' + L/ri²·f) = 0
                # Following Kore: kore-main/bin/assemble.py:1614-1630

                L = l * (l + 1)

                # Row 1: f(ri) = 0
                A[row_icb, :] .= zero(ComplexF64)
                B[row_icb, :] .= zero(ComplexF64)
                A[row_icb, block_range] = ComplexF64.(inner_vals)

                # Row 2: Em·(-f'' - (2/ri)·f' + (L/ri²)·f) = 0
                # We need to use the row BEFORE row_icb (row_icb-1) for the second BC
                # This replaces the last interior point
                row_icb2 = row_icb - 1

                # Zero out row
                A[row_icb2, :] .= zero(ComplexF64)
                B[row_icb2, :] .= zero(ComplexF64)
                value_term = (L / ri^2) .* inner_vals
                deriv1_term = -(2.0 / ri) .* inner_deriv
                deriv2_term = -inner_second
                A[row_icb2, block_range] = ComplexF64.(params.Em * (value_term + deriv1_term + deriv2_term))

            else
                # Simple conducting: f = 0 (no penetration)
                A[row_icb, :] .= zero(ComplexF64)
                B[row_icb, :] .= zero(ComplexF64)
                A[row_icb, block_range] = ComplexF64.(inner_vals)
            end
        end

    elseif section == :g  # Toroidal magnetic field
        for (k, l) in enumerate(op.ll_g)
            # Offset to g section: after u, v, f sections
            row_base = (nb_u + nb_v + nb_f + k - 1) * n_per_mode
            block_range = (row_base + 1):(row_base + n_per_mode)

            # ----------------------------------------------------------------
            # Outer boundary (CMB): g = 0 (for all BC types)
            # ----------------------------------------------------------------
            # Following Kore: kore-main/bin/assemble.py:1511-1522
            row_cmb = row_base + 1
            A[row_cmb, :] .= zero(ComplexF64)
            B[row_cmb, :] .= zero(ComplexF64)
            A[row_cmb, block_range] = ComplexF64.(outer_vals)

            # ----------------------------------------------------------------
            # Inner boundary (ICB)
            # ----------------------------------------------------------------
            row_icb = row_base + n_per_mode

            if params.bci_magnetic == 0
                # Insulating: g = 0
                A[row_icb, :] .= zero(ComplexF64)
                B[row_icb, :] .= zero(ComplexF64)
                A[row_icb, block_range] = ComplexF64.(inner_vals)

            elseif params.bci_magnetic == 1
                freq = params.forcing_frequency
                Em = params.Em
                if Em <= 0
                    error("Conducting magnetic BC requires Em > 0")
                end

                if iszero(freq)
                    A[row_icb, :] .= zero(ComplexF64)
                    B[row_icb, :] .= zero(ComplexF64)
                    A[row_icb, block_range] = ComplexF64.(r_inner .* inner_vals .- l .* inner_deriv)
                else
                    kwave = (1 - 1im) * sqrt(complex(freq) / (2 * Em))
                    dlog = spherical_bessel_j_logderiv(l, kwave * ri)

                    A[row_icb, :] .= zero(ComplexF64)
                    B[row_icb, :] .= zero(ComplexF64)
                    A[row_icb, block_range] = ComplexF64.(inner_vals) .-
                                             kwave * dlog .* ComplexF64.(inner_deriv)
                end

            elseif params.bci_magnetic == 2
                # Perfect conductor: Em·(-g' - 1/ri·g) = 0
                # Following Kore: kore-main/bin/assemble.py:1631-1641

                # Zero out row
                A[row_icb, :] .= zero(ComplexF64)
                B[row_icb, :] .= zero(ComplexF64)
                value_term = -(1.0 / ri) .* inner_vals
                deriv1_term = -inner_deriv
                A[row_icb, block_range] = ComplexF64.(params.Em * (value_term + deriv1_term))

            else
                # Default: g = 0
                A[row_icb, :] .= zero(ComplexF64)
                B[row_icb, :] .= zero(ComplexF64)
                A[row_icb, block_range] = ComplexF64.(inner_vals)
            end
        end
    end

    return nothing
end
