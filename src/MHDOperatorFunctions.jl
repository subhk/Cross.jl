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

For POLOIDAL field (section f):
- Insulating CMB: (l+1)·f(ro) + ro·f'(ro) = 0
- Insulating ICB: l·f(ri) - ri·f'(ri) = 0
- Conducting: f = 0 (no penetration)

For TOROIDAL field (section g):
- Insulating: g = 0 (no toroidal field outside)
- Conducting: g = 0 (different physics, same condition)
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

            else
                # Perfectly conducting: f = 0 (no penetration)
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

            # Outer boundary: g = 0 (for both insulating and conducting)
            # Following Kore: kore-main/bin/assemble.py:1511-1522
            row_cmb = row_base + 1
            A[row_cmb, :] .= 0.0
            B[row_cmb, :] .= 0.0
            for n in 0:N
                col = row_base + n + 1
                A[row_cmb, col] = 1.0  # T_n(1) = 1
            end

            # Inner boundary: g = 0 (for both insulating and conducting)
            row_icb = row_base + n_per_mode
            A[row_icb, :] .= 0.0
            B[row_icb, :] .= 0.0
            for n in 0:N
                col = row_base + n + 1
                A[row_icb, col] = (-1.0)^n  # T_n(-1) = (-1)^n
            end
        end
    end

    return nothing
end
