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

"""
    spherical_bessel_j_logderiv(l, x)

Compute the logarithmic derivative of the spherical Bessel function of the first kind:
    d/dx[log(j_l(x))] = j'_l(x) / j_l(x)

Uses the recurrence relation: j'_l(x) = l/x * j_l(x) - j_{l+1}(x)
Therefore: j'_l(x) / j_l(x) = l/x - j_{l+1}(x) / j_l(x)

This is numerically stable for complex arguments, which is needed for
conducting inner core boundary conditions with Bessel wavenumber k = (1-i)√(ω/(2Em)).

Following Kore's utils.py dlogjl function (lines 487-526).
"""
function spherical_bessel_j_logderiv(l::Int, x::Complex{T}) where {T<:Real}
    using SpecialFunctions

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
    L = l * (l + 1)

    # Lorentz force from background field
    # For axial field: involves h(r) = r and derivatives
    # Following Kore operators.py lines 240-250

    # Diagonal term: couples g at same l to u
    # Le² * L * (operators involving h(r))
    return Le^2 * L * (op.r1_h0_D0 + op.r2_h0_D1)
end

function operator_lorentz_poloidal_offdiag(op::MHDStabilityOperator{T},
                                           l::Int, m::Int, offset::Int,
                                           Le::T) where {T}
    # Off-diagonal Lorentz coupling (l±1)
    # Similar structure to Coriolis off-diagonal terms
    L = l * (l + 1)

    if offset == -1
        # Coupling from g at l to u at l-1
        C = sqrt((l^2 - m^2) * (l^2 - 1)) / (2l - 1)
        return Le^2 * C * (op.r2_h0_D0 - (l-1) * op.r1_h0_D0)
    elseif offset == 1
        # Coupling from g at l to u at l+1
        C = sqrt((l + m + 1) * (l - m + 1) * l * (l + 2)) / (2l + 3)
        return Le^2 * C * (op.r2_h0_D0 + (l+2) * op.r1_h0_D0)
    else
        error("offset must be ±1 for Lorentz off-diagonal")
    end
end

"""
    operator_lorentz_toroidal(op, l, Le)

Lorentz force acting on toroidal velocity from background magnetic field.
"""
function operator_lorentz_toroidal(op::MHDStabilityOperator{T},
                                   l::Int, Le::T) where {T}
    L = l * (l + 1)

    # Toroidal Lorentz force
    # Couples poloidal magnetic perturbation f to toroidal velocity v
    return Le^2 * L * op.r1_h0_D0
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
                                            l::Int) where {T}
    L = l * (l + 1)

    # Poloidal velocity advecting poloidal magnetic field
    # Following Kore operators.py lines 313-315
    return L * (op.r1_h0_D1 - op.r0_h0_D0)
end

function operator_induction_poloidal_from_v(op::MHDStabilityOperator{T},
                                            l::Int) where {T}
    # Toroidal velocity shearing background field
    # Creates poloidal magnetic field
    return -op.r1_h0_D0
end

"""
    operator_induction_toroidal(op, l, m)

Induction of toroidal magnetic field by velocity advection.
Implements Kore's induction equation for section g.
"""
function operator_induction_toroidal_from_u(op::MHDStabilityOperator{T},
                                           l::Int, m::Int, offset::Int) where {T}
    # Poloidal velocity shearing background field
    # Creates toroidal magnetic field at l±1

    if offset == -1
        C = sqrt((l^2 - m^2) * (l^2 - 1)) / (2l - 1)
        return -C * ((l - 1) * op.r0_h0_D0 + op.r1_h0_D1)
    elseif offset == 1
        C = sqrt((l + m + 1) * (l - m + 1) * l * (l + 2)) / (2l + 3)
        return C * ((l + 2) * op.r0_h0_D0 - op.r1_h0_D1)
    else
        error("offset must be ±1 for induction off-diagonal")
    end
end

function operator_induction_toroidal_from_v(op::MHDStabilityOperator{T},
                                            l::Int) where {T}
    L = l * (l + 1)

    # Toroidal velocity advecting toroidal field (diagonal)
    return L * op.r1_h0_D0
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

    # Diffusion: Em * ∇²B
    # For poloidal field (no-curl equation)
    # Following Kore operators.py lines 320-322
    return Em * (-L * op.r0_D0_f + 2 * op.r1_D1_f + op.r2_D2_f)
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

    # Toroidal magnetic diffusion
    # Following Kore operators.py lines 675-680
    # More terms than poloidal due to curl-curl in spherical coordinates
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
    return op.r2_D0_f
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
    return op.r2_D0_g
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

Apply boundary conditions to magnetic field sections.

Following Kore's implementation (kore-main/bin/assemble.py:1472-1786):

Boundary condition options (controlled by bci_magnetic/bco_magnetic):
- 0 = insulating
- 1 = conducting (finite conductivity, uses Bessel functions)
- 2 = perfect conductor

For POLOIDAL field (section f):
- Insulating CMB: (l+1)·f(ro) + ro·f'(ro) = 0
- Insulating ICB: l·f(ri) - ri·f'(ri) = 0
- Conducting ICB: f - k·dlogjl(l, k·ri)·f' = 0, where k = (1-i)√(ω/(2Em))
- Perfect conductor ICB: f = 0 and Em·(-f'' - 2/ri·f' + L/ri²·f) = 0 (2 rows)
- Conducting/Perfect CMB: f = 0 (no penetration)

For TOROIDAL field (section g):
- Insulating: g = 0 (no toroidal field outside)
- Conducting: g = 0 (different physics, same condition)
- Perfect conductor ICB: Em·(-g' - 1/ri·g) = 0
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

    if section == :f  # Poloidal magnetic field
        for (k, l) in enumerate(op.ll_f)
            # Offset to f section: after u and v sections
            row_base = (nb_u + nb_v + k - 1) * n_per_mode

            # ----------------------------------------------------------------
            # Outer boundary (CMB, r = ro = 1)
            # ----------------------------------------------------------------
            row_cmb = row_base + 1

            if params.bco_magnetic == 0
                # Insulating CMB: (l+1)·f(ro) + ro·f'(ro) = 0
                # Following Kore: kore-main/bin/assemble.py:1494-1509

                # Zero out row
                A[row_cmb, :] .= 0.0
                B[row_cmb, :] .= 0.0

                # Build constraint: (l+1)·f + ro·f'
                # f evaluated at boundary (Chebyshev at x=1)
                for n in 0:N
                    col = row_base + n + 1
                    Tn_at_1 = 1.0  # T_n(1) = 1 for all n
                    A[row_cmb, col] = (l + 1) * Tn_at_1
                end

                # f' at boundary: use derivative operator
                # Get first derivative operator (from UltrasphericalSpectral module)
                D1 = UltrasphericalSpectral.sparse_radial_operator(0, 1, N, ri, ro)

                # Add ro·f' term (ro = 1)
                for n in 0:N
                    col = row_base + n + 1
                    A[row_cmb, col] += ro * D1[1, n+1]  # First row = outer boundary
                end

            else
                # Perfectly conducting: f = 0 (no penetration)
                A[row_cmb, :] .= 0.0
                B[row_cmb, :] .= 0.0
                for n in 0:N
                    col = row_base + n + 1
                    A[row_cmb, col] = 1.0  # T_n(1) = 1
                end
            end

            # ----------------------------------------------------------------
            # Inner boundary (ICB, r = ri)
            # ----------------------------------------------------------------
            row_icb = row_base + n_per_mode

            if params.bci_magnetic == 0
                # Insulating ICB: l·f(ri) - ri·f'(ri) = 0
                # Following Kore: kore-main/bin/assemble.py:1548-1572

                # Zero out row
                A[row_icb, :] .= 0.0
                B[row_icb, :] .= 0.0

                # Build constraint: l·f - ri·f'
                # f evaluated at ICB (Chebyshev at x=-1)
                for n in 0:N
                    col = row_base + n + 1
                    Tn_at_minus1 = (-1.0)^n  # T_n(-1) = (-1)^n
                    A[row_icb, col] = l * Tn_at_minus1
                end

                # f' at boundary: use derivative operator (from UltrasphericalSpectral module)
                D1 = UltrasphericalSpectral.sparse_radial_operator(0, 1, N, ri, ro)

                # Subtract ri·f' term (note the MINUS sign, different from CMB!)
                for n in 0:N
                    col = row_base + n + 1
                    A[row_icb, col] -= ri * D1[N+1, n+1]  # Last row = inner boundary
                end

            elseif params.bci_magnetic == 1
                # Conducting ICB (finite conductivity): f - k·dlogjl(l, k·ri)·f' = 0
                # where k = (1-i)√(forcing_frequency/(2Em))
                # Following Kore: kore-main/bin/assemble.py:1575-1612
                # NOTE: This creates a nonlinear eigenvalue problem when frequency = eigenvalue σ
                # For now, requires forcing_frequency parameter to be set

                error("Conducting ICB with finite conductivity not yet fully implemented. " *
                      "Use bci_magnetic=0 (insulating) or bci_magnetic=2 (perfect conductor)")

            elseif params.bci_magnetic == 2
                # Perfect conductor ICB: 2-row boundary condition
                # Row 1: f = 0
                # Row 2: Em·(-f'' - 2/ri·f' + L/ri²·f) = 0
                # Following Kore: kore-main/bin/assemble.py:1614-1630

                L = l * (l + 1)

                # Row 1: f(ri) = 0
                A[row_icb, :] .= 0.0
                B[row_icb, :] .= 0.0
                for n in 0:N
                    col = row_base + n + 1
                    Tn_at_minus1 = (-1.0)^n  # T_n(-1) = (-1)^n
                    A[row_icb, col] = Tn_at_minus1
                end

                # Row 2: Em·(-f'' - (2/ri)·f' + (L/ri²)·f) = 0
                # We need to use the row BEFORE row_icb (row_icb-1) for the second BC
                # This replaces the last interior point
                row_icb2 = row_icb - 1

                # Zero out row
                A[row_icb2, :] .= 0.0
                B[row_icb2, :] .= 0.0

                # Get derivative operators
                D1 = UltrasphericalSpectral.sparse_radial_operator(0, 1, N, ri, ro)
                D2 = UltrasphericalSpectral.sparse_radial_operator(0, 2, N, ri, ro)

                # Em·(-f'' - (2/ri)·f' + (L/ri²)·f) at ICB
                for n in 0:N
                    col = row_base + n + 1
                    Tn_at_minus1 = (-1.0)^n

                    # Value term: (L/ri²)·f
                    value_term = (L / ri^2) * Tn_at_minus1

                    # First derivative term: -(2/ri)·f'
                    deriv1_term = -(2.0 / ri) * D1[N+1, n+1]

                    # Second derivative term: -f''
                    deriv2_term = -D2[N+1, n+1]

                    A[row_icb2, col] = params.Em * (value_term + deriv1_term + deriv2_term)
                end

            else
                # Simple conducting: f = 0 (no penetration)
                A[row_icb, :] .= 0.0
                B[row_icb, :] .= 0.0
                for n in 0:N
                    col = row_base + n + 1
                    A[row_icb, col] = (-1.0)^n  # T_n(-1) = (-1)^n
                end
            end
        end

    elseif section == :g  # Toroidal magnetic field
        for (k, l) in enumerate(op.ll_g)
            # Offset to g section: after u, v, f sections
            row_base = (nb_u + nb_v + nb_f + k - 1) * n_per_mode

            # ----------------------------------------------------------------
            # Outer boundary (CMB): g = 0 (for all BC types)
            # ----------------------------------------------------------------
            # Following Kore: kore-main/bin/assemble.py:1511-1522
            row_cmb = row_base + 1
            A[row_cmb, :] .= 0.0
            B[row_cmb, :] .= 0.0
            for n in 0:N
                col = row_base + n + 1
                A[row_cmb, col] = 1.0  # T_n(1) = 1
            end

            # ----------------------------------------------------------------
            # Inner boundary (ICB)
            # ----------------------------------------------------------------
            row_icb = row_base + n_per_mode

            if params.bci_magnetic == 0 || params.bci_magnetic == 1
                # Insulating or conducting: g = 0
                A[row_icb, :] .= 0.0
                B[row_icb, :] .= 0.0
                for n in 0:N
                    col = row_base + n + 1
                    A[row_icb, col] = (-1.0)^n  # T_n(-1) = (-1)^n
                end

            elseif params.bci_magnetic == 2
                # Perfect conductor: Em·(-g' - 1/ri·g) = 0
                # Following Kore: kore-main/bin/assemble.py:1631-1641

                # Zero out row
                A[row_icb, :] .= 0.0
                B[row_icb, :] .= 0.0

                # Get first derivative operator
                D1 = UltrasphericalSpectral.sparse_radial_operator(0, 1, N, ri, ro)

                # Em·(-g' - (1/ri)·g) at ICB
                for n in 0:N
                    col = row_base + n + 1
                    Tn_at_minus1 = (-1.0)^n

                    # Value term: -(1/ri)·g
                    value_term = -(1.0 / ri) * Tn_at_minus1

                    # First derivative term: -g'
                    deriv1_term = -D1[N+1, n+1]

                    A[row_icb, col] = params.Em * (value_term + deriv1_term)
                end

            else
                # Default: g = 0
                A[row_icb, :] .= 0.0
                B[row_icb, :] .= 0.0
                for n in 0:N
                    col = row_base + n + 1
                    A[row_icb, col] = (-1.0)^n  # T_n(-1) = (-1)^n
                end
            end
        end
    end

    return nothing
end
