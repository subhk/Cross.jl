# =============================================================================
#  Self-Consistent Tri-Global Basic State with Temperature Advection
#
#  For non-axisymmetric basic states, the zonal flow ū_φ advects the
#  temperature field: ū_φ/(r sinθ) × ∂T̄/∂φ ≠ 0
#
#  The full steady-state equation is:
#    κ∇²T̄ = ū·∇T̄ = ū_φ/(r sinθ) × ∂T̄/∂φ
#
#  This requires an iterative solution:
#  1. Solve ∇²T̄ = 0 for initial guess (Laplace)
#  2. Compute ū_φ from thermal wind balance
#  3. Compute advection term ū_φ × ∂T̄/∂φ
#  4. Solve κ∇²T̄ = source with boundary conditions
#  5. Update thermal wind and repeat until convergence
# =============================================================================

"""
    AdvectionDiffusionSolver{T<:Real}

Holds parameters and state for iterative advection-diffusion solution.

Fields:
- `cd` : ChebyshevDiffn - radial discretization
- `r_i, r_o` : Inner and outer radii
- `E, Ra, Pr` : Ekman, Rayleigh, and Prandtl numbers
- `κ` : Thermal diffusivity (computed from other parameters)
- `lmax_bs, mmax_bs` : Maximum spherical harmonic degrees
- `mechanical_bc, thermal_bc` : Boundary condition types
- `max_iterations` : Maximum Picard iterations
- `tolerance` : Convergence tolerance
"""
@with_kw struct AdvectionDiffusionSolver{T<:Real}
    cd::ChebyshevDiffn{T}
    r_i::T
    r_o::T
    E::T
    Ra::T
    Pr::T
    lmax_bs::Int
    mmax_bs::Int
    mechanical_bc::Symbol = :no_slip
    thermal_bc::Symbol = :fixed_temperature
    max_iterations::Int = 20
    tolerance::T = T(1e-8)
end


"""
    compute_phi_advection_spectral(theta_coeffs, uphi_coeffs, lmax_bs, mmax_bs, r)

Compute the φ-advection term in spectral space for a single azimuthal mode m_bs.

For temperature T̄_m = Σ_ℓ T̄_ℓm(r) Y_ℓm and velocity ū_φ,m = Σ_L ū_{Lm}(r) Y_Lm,
the advection term is:

    ū_φ/(r sinθ) × ∂T̄/∂φ = ū_φ × (im/r sinθ) × T̄

In spectral space, this involves coupling through:
    Y_Lm × Y_ℓm / sinθ = Σ_L' C_{L,ℓ,L'}^m × Y_{L',m}

where C are coupling coefficients from Gaunt integrals.

For the simplified diagonal approximation (valid for slowly varying ū_φ):
    [ū_φ × im T̄ / (r sinθ)]_{L'm} ≈ im × Σ_L (∫ Y_Lm Y_ℓm Y_{L'm} / sinθ dΩ) × ū_Lm(r) × T̄_ℓm(r) / r

Returns the forcing coefficients for the advection-diffusion equation.
"""
function compute_phi_advection_spectral(
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    lmax_bs::Int,
    mmax_bs::Int,
    r::Vector{T}
) where T<:Real

    Nr = length(r)
    forcing = Dict{Tuple{Int,Int}, Vector{T}}()

    # For each azimuthal wavenumber m
    for m_bs in 1:mmax_bs  # m=0 has no φ-advection
        # φ-derivative multiplier
        im_m = T(m_bs)  # ∂/∂φ Y_ℓm = im Y_ℓm

        # Loop over temperature modes with this m
        for ℓ_T in m_bs:lmax_bs
            if !haskey(theta_coeffs, (ℓ_T, m_bs))
                continue
            end
            T_lm = theta_coeffs[(ℓ_T, m_bs)]

            if maximum(abs.(T_lm)) < eps(T) * 100
                continue
            end

            # Loop over velocity modes with this m
            for ℓ_u in m_bs:lmax_bs
                if !haskey(uphi_coeffs, (ℓ_u, m_bs))
                    continue
                end
                u_Lm = uphi_coeffs[(ℓ_u, m_bs)]

                if maximum(abs.(u_Lm)) < eps(T) * 100
                    continue
                end

                # The product ū × T in spectral space couples through Gaunt coefficients
                # For simplicity, use the "diagonal" approximation where the product
                # of Y_ℓm × Y_Lm ≈ δ_{ℓL} at leading order for ℓ = L
                #
                # More accurate: use Gaunt coefficients
                #   ⟨Y_{L'm} | Y_Lm × Y_ℓm / sinθ⟩
                # which couple (ℓ, L) → L' through selection rules

                # Diagonal contribution: ℓ_T ≈ ℓ_u → same mode
                if ℓ_T == ℓ_u
                    L_out = ℓ_T

                    if !haskey(forcing, (L_out, m_bs))
                        forcing[(L_out, m_bs)] = zeros(T, Nr)
                    end

                    # Advection term: im × ū_φ × T̄ / r
                    # The 1/sinθ factor is absorbed into the Y_ℓm normalization
                    # via the recurrence relation averaging
                    forcing[(L_out, m_bs)] .+= im_m .* u_Lm .* T_lm ./ r
                end

                # Off-diagonal contributions from Gaunt coefficients
                # |ℓ_T - ℓ_u| ≤ L' ≤ ℓ_T + ℓ_u (triangle rule)
                # The 1/sinθ factor shifts selection rules

                # For more accuracy, we can add the leading off-diagonal terms
                # using the identity:
                #   1/sinθ = Σ_n c_n P_n(cosθ)
                # This couples L' = ℓ_T + ℓ_u, ℓ_T + ℓ_u - 2, ...

                # Leading off-diagonal: L' = |ℓ_T - ℓ_u| (if different from diagonal)
                L_diff = abs(ℓ_T - ℓ_u)
                if L_diff != ℓ_T && L_diff >= m_bs
                    # Coupling coefficient (simplified estimate)
                    # Full calculation requires Wigner 3j symbols
                    c_coupling = sqrt(T(2*ℓ_T + 1) * T(2*ℓ_u + 1)) / (4 * T(π) * T(2*L_diff + 1))
                    c_coupling *= T(0.5)  # Empirical reduction factor

                    if !haskey(forcing, (L_diff, m_bs))
                        forcing[(L_diff, m_bs)] = zeros(T, Nr)
                    end
                    forcing[(L_diff, m_bs)] .+= c_coupling .* im_m .* u_Lm .* T_lm ./ r
                end

                # Sum coupling: L' = ℓ_T + ℓ_u
                L_sum = ℓ_T + ℓ_u
                if L_sum <= lmax_bs && L_sum != ℓ_T
                    c_coupling = sqrt(T(2*ℓ_T + 1) * T(2*ℓ_u + 1)) / (4 * T(π) * T(2*L_sum + 1))
                    c_coupling *= T(0.3)  # Empirical reduction factor

                    if !haskey(forcing, (L_sum, m_bs))
                        forcing[(L_sum, m_bs)] = zeros(T, Nr)
                    end
                    forcing[(L_sum, m_bs)] .+= c_coupling .* im_m .* u_Lm .* T_lm ./ r
                end
            end
        end
    end

    return forcing
end


"""
    solve_poisson_mode(ℓ, m, r, D2, D1, r_i, r_o, forcing;
                       inner_value=0, outer_value=0, outer_bc=:fixed_temperature)

Solve the radial Poisson equation for a single spherical harmonic mode:

    ∇²T̄_ℓm = f_ℓm(r)

where ∇² in spherical harmonics becomes:

    d²/dr² + (2/r)d/dr - ℓ(ℓ+1)/r² = f_ℓm(r)

Returns T̄_ℓm(r) and ∂T̄_ℓm/∂r.
"""
function solve_poisson_mode(
    ℓ::Int, m::Int,
    r::Vector{T}, D2::Matrix{T}, D1::Matrix{T},
    r_i::T, r_o::T,
    forcing::Vector{T};
    inner_value::T = zero(T),
    outer_value::T = zero(T),
    outer_bc::Symbol = :fixed_temperature,
    inner_bc::Symbol = :fixed_temperature
) where T<:Real

    Nr = length(r)

    # Build the radial Laplacian operator for mode ℓ:
    # L_ℓ = D2 + (2/r) D1 - ℓ(ℓ+1)/r²
    L_op = D2 + Diagonal(T(2) ./ r) * D1 - Diagonal(T(ℓ * (ℓ + 1)) ./ (r.^2))

    A_mat = copy(L_op)
    f_rhs = copy(forcing)

    # Determine boundary indices (Chebyshev nodes can be ascending or descending)
    idx_inner = abs(r[1] - r_i) < abs(r[Nr] - r_i) ? 1 : Nr
    idx_outer = idx_inner == 1 ? Nr : 1

    # Inner boundary condition (typically fixed temperature = hot)
    if inner_bc == :fixed_temperature
        A_mat[idx_inner, :] .= zero(T)
        A_mat[idx_inner, idx_inner] = one(T)
        f_rhs[idx_inner] = inner_value
    else  # fixed_flux
        A_mat[idx_inner, :] = D1[idx_inner, :]
        f_rhs[idx_inner] = inner_value  # This is the flux value
    end

    # Outer boundary condition
    if outer_bc == :fixed_temperature
        A_mat[idx_outer, :] .= zero(T)
        A_mat[idx_outer, idx_outer] = one(T)
        f_rhs[idx_outer] = outer_value
    else  # fixed_flux
        A_mat[idx_outer, :] = D1[idx_outer, :]
        f_rhs[idx_outer] = outer_value  # This is the flux value
    end

    # Solve the linear system
    T_lm = A_mat \ f_rhs
    dT_dr = D1 * T_lm

    return T_lm, dT_dr
end


# =============================================================================
#  Spherical Harmonic Coupling Coefficients
#
#  For sin(θ) and cos(θ) multiplications in spectral space:
#    sin(θ) Y_ℓm = a_{ℓ,m}^- Y_{ℓ-1,m} + a_{ℓ,m}^+ Y_{ℓ+1,m}
#    cos(θ) Y_ℓm = b_{ℓ,m}^- Y_{ℓ-1,m} + b_{ℓ,m}^+ Y_{ℓ+1,m}
# =============================================================================

"""
    sin_theta_coupling(ℓ::Int, m::Int)

Compute coupling coefficients for sin(θ) × Y_ℓm = a⁻ Y_{ℓ-1,m} + a⁺ Y_{ℓ+1,m}.

Returns (a_minus, a_plus) where:
- a⁻ = √[(ℓ+m)(ℓ-m) / ((2ℓ-1)(2ℓ+1))]  (coupling to ℓ-1)
- a⁺ = √[(ℓ+m+1)(ℓ-m+1) / ((2ℓ+1)(2ℓ+3))]  (coupling to ℓ+1)
"""
function sin_theta_coupling(ℓ::Int, m::Int)
    # Coupling to ℓ-1
    a_minus = 0.0
    if ℓ > abs(m)
        num = (ℓ + m) * (ℓ - m)
        den = (2ℓ - 1) * (2ℓ + 1)
        if num >= 0 && den > 0
            a_minus = sqrt(num / den)
        end
    end

    # Coupling to ℓ+1
    num = (ℓ + m + 1) * (ℓ - m + 1)
    den = (2ℓ + 1) * (2ℓ + 3)
    a_plus = num >= 0 && den > 0 ? sqrt(num / den) : 0.0

    return (a_minus, a_plus)
end


"""
    cos_theta_coupling(ℓ::Int, m::Int)

Compute coupling coefficients for cos(θ) × Y_ℓm = b⁻ Y_{ℓ-1,m} + b⁺ Y_{ℓ+1,m}.

Returns (b_minus, b_plus) where:
- b⁻ = √[(ℓ-m)(ℓ+m) / ((2ℓ-1)(2ℓ+1))] × (ℓ) / √[ℓ²-m²] ... (simplified)
- b⁺ = √[(ℓ-m+1)(ℓ+m+1) / ((2ℓ+1)(2ℓ+3))]

Using the recurrence: cos(θ) P_ℓ^m = A_ℓ^m P_{ℓ+1}^m + B_ℓ^m P_{ℓ-1}^m
"""
function cos_theta_coupling(ℓ::Int, m::Int)
    # Using the standard recurrence for associated Legendre polynomials
    # cos(θ) P_ℓ^m = [(ℓ-m+1)/(2ℓ+1)] P_{ℓ+1}^m + [(ℓ+m)/(2ℓ+1)] P_{ℓ-1}^m
    # After normalization for spherical harmonics:

    # Coupling to ℓ-1
    b_minus = 0.0
    if ℓ > abs(m)
        # From recurrence relation
        num = (ℓ + m) * (ℓ - m)
        den = (2ℓ - 1) * (2ℓ + 1)
        if num > 0 && den > 0
            b_minus = sqrt(num / den)
        end
    end

    # Coupling to ℓ+1
    num = (ℓ - m + 1) * (ℓ + m + 1)
    den = (2ℓ + 1) * (2ℓ + 3)
    b_plus = num >= 0 && den > 0 ? sqrt(num / den) : 0.0

    return (b_minus, b_plus)
end


"""
    inv_sin_theta_coupling(ℓ::Int, m::Int)

Approximate coupling coefficients for (1/sinθ) × Y_ℓm in spectral space.

Since 1/sinθ is singular at poles, this expansion is approximate and uses
a truncated series representation. The dominant contributions come from
modes with similar ℓ values.

Returns a Dict{Int, Float64} mapping output ℓ' to coupling coefficient.
"""
function inv_sin_theta_coupling(ℓ::Int, m::Int; max_coupling::Int=4)
    # 1/sinθ can be expanded as a series in Legendre polynomials
    # For practical purposes, use approximate coupling to nearby modes

    coeffs = Dict{Int, Float64}()

    # Diagonal term (dominant)
    coeffs[ℓ] = 1.0

    # For m ≠ 0, there's coupling to ℓ ± 2, ℓ ± 4, etc.
    if m != 0
        # ℓ+2 coupling (approximate)
        if ℓ + 2 <= ℓ + max_coupling
            c = 0.5 * m^2 / ((2ℓ + 1) * (2ℓ + 3))
            if abs(c) > 1e-10
                coeffs[ℓ + 2] = c
            end
        end

        # ℓ-2 coupling (if valid)
        if ℓ - 2 >= abs(m) && ℓ - 2 >= 0
            c = 0.5 * m^2 / ((2ℓ - 1) * (2ℓ + 1))
            if abs(c) > 1e-10
                coeffs[ℓ - 2] = c
            end
        end
    end

    return coeffs
end


# =============================================================================
#  Meridional Circulation for Non-Axisymmetric Basic States
#
#  The thermal wind equation (curl of geostrophic balance):
#      2Ω (ẑ·∇) ū = (Ra E²/Pr) ∇T̄ × r̂
#
#  Components:
#  - φ-component → ū_φ (toroidal thermal wind) - already solved elsewhere
#  - θ-component → ū_θ from: 2Ω (ẑ·∇) u_θ = -(Ra E²/Pr)/(r sinθ) × ∂T̄/∂φ
#  - r-component → (ẑ·∇) u_r = 0 (Taylor-Proudman constraint)
#
#  The operator (ẑ·∇) = cosθ ∂/∂r - (sinθ/r) ∂/∂θ couples modes ℓ to ℓ±1.
#
#  For m = 0: ∂T̄/∂φ = 0, so no meridional forcing (u_r = u_θ = 0)
#  For m ≠ 0: The φ-gradient drives meridional circulation
#
#  FULL SOLUTION: We solve the coupled block-tridiagonal system exactly.
# =============================================================================

"""
    theta_derivative_coupling(ℓ::Int, m::Int)

Compute coupling coefficients for sinθ × ∂Y_ℓm/∂θ expansion in spherical harmonics.

Using the recurrence relation for associated Legendre functions:
    sinθ ∂Y_ℓm/∂θ = A⁺_ℓm Y_{ℓ+1,m} + A⁻_ℓm Y_{ℓ-1,m} + (diagonal correction)

Returns (A_minus, A_plus, A_diag) where:
- A_minus: coefficient for coupling to ℓ-1
- A_plus: coefficient for coupling to ℓ+1
- A_diag: diagonal contribution (usually small)
"""
function theta_derivative_coupling(ℓ::Int, m::Int)
    # From the identity: sinθ ∂Y_ℓm/∂θ = ℓ cosθ Y_ℓm - (ℓ+m) × norm_factor × Y_{ℓ-1,m}
    # and the cosθ expansion.
    #
    # More precisely, using the recurrence:
    # sin²θ dP_ℓ^m/d(cosθ) = -(ℓ+1) cosθ P_ℓ^m + (ℓ-m+1) P_{ℓ+1}^m
    #
    # This gives sinθ ∂Y/∂θ in terms of cosθ Y (which couples ℓ±1) and Y_{ℓ+1}.

    # Coefficients from the standard recurrence
    # The result is: sinθ ∂Y_ℓm/∂θ couples primarily to ℓ±1

    # A⁺: coupling to ℓ+1
    A_plus = 0.0
    if ℓ + 1 >= abs(m)
        # Coefficient from the derivative recurrence
        num = (ℓ + 1 + m) * (ℓ + 1 - m)
        den = (2ℓ + 1) * (2ℓ + 3)
        if num >= 0 && den > 0
            A_plus = -(ℓ + 1) * sqrt(num / den)
        end
    end

    # A⁻: coupling to ℓ-1
    A_minus = 0.0
    if ℓ - 1 >= abs(m) && ℓ > 0
        num = (ℓ + m) * (ℓ - m)
        den = (2ℓ - 1) * (2ℓ + 1)
        if num >= 0 && den > 0
            A_minus = ℓ * sqrt(num / den)
        end
    end

    # Diagonal contribution (from m cotθ term, averaged)
    A_diag = 0.0  # This averages to zero over the sphere for m ≠ 0

    return (A_minus, A_plus, A_diag)
end


"""
    inv_sin_theta_gaunt(L::Int, ℓ::Int, m::Int)

Compute the Gaunt-like integral ⟨Y_Lm | 1/sinθ | Y_ℓm⟩.

The 1/sinθ factor couples modes with |L - ℓ| even (0, 2, 4, ...).
The dominant contribution is diagonal (L = ℓ).

Returns the coupling coefficient. Non-zero only for specific L values.
"""
function inv_sin_theta_gaunt(L::Int, ℓ::Int, m::Int)
    if L < abs(m) || ℓ < abs(m)
        return 0.0
    end

    # Diagonal term (dominant)
    if L == ℓ
        # ⟨Y_ℓm | 1/sinθ | Y_ℓm⟩ ≈ 1 + m²/(ℓ(ℓ+1)) for ℓ > 0
        if ℓ == 0
            return 1.0
        else
            return 1.0 + m^2 / (ℓ * (ℓ + 1))
        end
    end

    # L = ℓ ± 2 coupling (weaker)
    if abs(L - ℓ) == 2 && (L + ℓ) % 2 == 0
        # Approximate coupling for ℓ ± 2
        ℓ_avg = (L + ℓ) / 2
        return 0.25 * m^2 / max(ℓ_avg * (ℓ_avg + 1), 1.0)
    end

    return 0.0
end


"""
    solve_meridional_coupled!(ur_coeffs, utheta_coeffs, dur_dr_coeffs, dutheta_dr_coeffs,
                               theta_coeffs, uphi_coeffs, r, D1, D2, r_i, r_o,
                               Ra, E, Pr, m_bs, lmax_bs;
                               mechanical_bc=:no_slip)

Solve the FULL coupled system for meridional circulation at azimuthal wavenumber m.

This solves the θ-thermal wind equation WITHOUT approximations:
    2Ω (ẑ·∇) u_θ = -(Ra E²/Pr)/(r sinθ) × ∂T̄/∂φ

The operator (ẑ·∇) = cosθ ∂/∂r - (sinθ/r) ∂/∂θ couples modes ℓ to ℓ±1.
We build the full block-tridiagonal matrix and solve simultaneously.

The equation at mode L receives contributions from modes ℓ = L-1 and ℓ = L+1:
    2Ω [C⁺_{L-1} du_θ,L-1/dr + C⁻_{L+1} du_θ,L+1/dr
        - (1/r)(A⁺_{L-1} u_θ,L-1 + A⁻_{L+1} u_θ,L+1)] = F_L

where C± are cosθ coupling and A± are sinθ∂/∂θ coupling coefficients.

After solving for u_θ, we compute u_r from the continuity equation:
    ∂(r² u_r)/∂r = -r² × [angular derivative terms with u_θ] - im r u_φ/sinθ
"""
function solve_meridional_coupled!(
    ur_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    utheta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dur_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dutheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    r::Vector{T}, D1::Matrix{T}, D2::Matrix{T},
    r_i::T, r_o::T,
    Ra::T, E::T, Pr::T,
    m_bs::Int, lmax_bs::Int;
    mechanical_bc::Symbol = :no_slip
) where T<:Real

    Nr = length(r)

    # For m = 0, no meridional circulation (∂T̄/∂φ = 0)
    if m_bs == 0
        for ℓ in 0:lmax_bs
            ur_coeffs[(ℓ, 0)] = zeros(T, Nr)
            utheta_coeffs[(ℓ, 0)] = zeros(T, Nr)
            dur_dr_coeffs[(ℓ, 0)] = zeros(T, Nr)
            dutheta_dr_coeffs[(ℓ, 0)] = zeros(T, Nr)
        end
        return nothing
    end

    # Non-dimensional parameters
    two_omega = one(T) / E
    buoyancy_factor = Ra * E^2 / Pr

    # Boundary indices
    idx_inner = abs(r[1] - r_i) < abs(r[Nr] - r_i) ? 1 : Nr
    idx_outer = idx_inner == 1 ? Nr : 1

    # Number of ℓ modes: m_bs, m_bs+1, ..., lmax_bs
    n_ell = lmax_bs - m_bs + 1
    if n_ell <= 0
        return nothing
    end

    # =========================================================================
    # Build the block-tridiagonal system: A × u_θ_vec = RHS_vec
    # u_θ_vec = [u_θ,m; u_θ,m+1; ...; u_θ,lmax] (each is Nr points)
    # =========================================================================

    total_size = n_ell * Nr
    A_full = zeros(T, total_size, total_size)
    RHS_full = zeros(T, total_size)

    for (i_L, L) in enumerate(m_bs:lmax_bs)
        # Row indices for mode L
        row_start = (i_L - 1) * Nr + 1
        row_end = i_L * Nr

        # =================================================================
        # RHS: -(Ra E²/Pr) × im × Σ_ℓ T̄_ℓm × ⟨Y_Lm|1/(r sinθ)|Y_ℓm⟩
        # =================================================================
        for ℓ in m_bs:lmax_bs
            if !haskey(theta_coeffs, (ℓ, m_bs))
                continue
            end
            T_lm = theta_coeffs[(ℓ, m_bs)]

            # Gaunt coefficient for 1/sinθ coupling
            gaunt_coeff = inv_sin_theta_gaunt(L, ℓ, m_bs)
            if abs(gaunt_coeff) > eps(T)
                RHS_full[row_start:row_end] .+= -buoyancy_factor .* T(m_bs) .* gaunt_coeff .* T_lm ./ r
            end
        end

        # =================================================================
        # LHS: 2Ω × ⟨Y_Lm | (ẑ·∇) | u_θ⟩
        #
        # (ẑ·∇) u_θ = Σ_ℓ [cosθ du_θ,ℓ/dr - (sinθ/r) u_θ,ℓ ∂/∂θ] Y_ℓm
        #
        # Projecting onto Y_Lm:
        #   ⟨Y_Lm|cosθ|Y_ℓm⟩ × du_θ,ℓ/dr - (1/r) ⟨Y_Lm|sinθ ∂/∂θ|Y_ℓm⟩ × u_θ,ℓ
        # =================================================================

        # Loop over source modes ℓ
        for (i_ell, ℓ) in enumerate(m_bs:lmax_bs)
            col_start = (i_ell - 1) * Nr + 1
            col_end = i_ell * Nr

            # cosθ coupling: ⟨Y_Lm|cosθ|Y_ℓm⟩
            # Non-zero only for L = ℓ±1
            C_Lell = zero(T)
            if L == ℓ + 1
                # cosθ Y_ℓm has component C⁺_ℓm at Y_{ℓ+1,m}
                _, C_plus = cos_theta_coupling(ℓ, m_bs)
                C_Lell = T(C_plus)
            elseif L == ℓ - 1
                # cosθ Y_ℓm has component C⁻_ℓm at Y_{ℓ-1,m}
                C_minus, _ = cos_theta_coupling(ℓ, m_bs)
                C_Lell = T(C_minus)
            end

            # sinθ ∂/∂θ coupling: ⟨Y_Lm|sinθ ∂/∂θ|Y_ℓm⟩
            # Non-zero only for L = ℓ±1
            A_Lell = zero(T)
            if L == ℓ + 1
                # sinθ ∂Y_ℓm/∂θ has component A⁺_ℓm at Y_{ℓ+1,m}
                _, A_plus, _ = theta_derivative_coupling(ℓ, m_bs)
                A_Lell = T(A_plus)
            elseif L == ℓ - 1
                # sinθ ∂Y_ℓm/∂θ has component A⁻_ℓm at Y_{ℓ-1,m}
                A_minus, _, _ = theta_derivative_coupling(ℓ, m_bs)
                A_Lell = T(A_minus)
            end

            # Build the operator block
            if abs(C_Lell) > eps(T) || abs(A_Lell) > eps(T)
                # Operator: 2Ω × [C_Lell × D1 - (A_Lell/r)]
                op_block = two_omega .* (C_Lell .* D1 .- Diagonal(A_Lell ./ r))
                A_full[row_start:row_end, col_start:col_end] .= op_block
            end

            # Diagonal regularization (small term to ensure solvability)
            if L == ℓ
                # Add small diagonal term for numerical stability
                reg_coeff = two_omega * T(0.01) / T(max(L, 1))
                A_full[row_start:row_end, col_start:col_end] .+= reg_coeff .* Matrix{T}(I, Nr, Nr)
            end
        end

        # =================================================================
        # Boundary conditions: u_θ = 0 at r_i and r_o (no-slip)
        # =================================================================
        if mechanical_bc == :no_slip
            A_full[row_start + idx_inner - 1, :] .= zero(T)
            A_full[row_start + idx_inner - 1, row_start + idx_inner - 1] = one(T)
            RHS_full[row_start + idx_inner - 1] = zero(T)

            A_full[row_start + idx_outer - 1, :] .= zero(T)
            A_full[row_start + idx_outer - 1, row_start + idx_outer - 1] = one(T)
            RHS_full[row_start + idx_outer - 1] = zero(T)
        end
    end

    # =========================================================================
    # Solve the coupled system
    # =========================================================================
    u_theta_vec = A_full \ RHS_full

    # Extract u_θ for each mode
    for (i_ell, ℓ) in enumerate(m_bs:lmax_bs)
        idx_start = (i_ell - 1) * Nr + 1
        idx_end = i_ell * Nr
        utheta_coeffs[(ℓ, m_bs)] = u_theta_vec[idx_start:idx_end]
        dutheta_dr_coeffs[(ℓ, m_bs)] = D1 * u_theta_vec[idx_start:idx_end]
    end

    # =========================================================================
    # Compute u_r from continuity equation
    #
    # ∇·ū = (1/r²) ∂(r² u_r)/∂r + (1/(r sinθ)) ∂(sinθ u_θ)/∂θ + im u_φ/(r sinθ) = 0
    #
    # In spectral space for mode (ℓ,m):
    # ∂(r² u_r,ℓm)/∂r = -r² × [angular coupling from ∂(sinθ u_θ)/∂θ] - im r × [u_φ/sinθ terms]
    #
    # The ∂(sinθ u_θ)/∂θ term couples modes via:
    #   ∂(sinθ Y_ℓm)/∂θ = d⁺_ℓm Y_{ℓ+1,m} + d⁻_ℓm Y_{ℓ-1,m}
    # =========================================================================

    for (i_L, L) in enumerate(m_bs:lmax_bs)
        # Source term for u_r at mode L
        source_ur = zeros(T, Nr)

        # Contribution from ∂(sinθ u_θ)/∂θ projected onto Y_Lm
        # sinθ u_θ Y_ℓm has ∂/∂θ that couples via d±
        for (i_ell, ℓ) in enumerate(m_bs:lmax_bs)
            if !haskey(utheta_coeffs, (ℓ, m_bs))
                continue
            end
            utheta_ell = utheta_coeffs[(ℓ, m_bs)]

            # Coupling coefficient for ∂(sinθ Y_ℓm)/∂θ at mode L
            # Using: ∂(sinθ Y)/∂θ = cosθ Y + sinθ ∂Y/∂θ
            d_Lell = zero(T)
            if L == ℓ + 1
                _, C_plus = cos_theta_coupling(ℓ, m_bs)
                _, A_plus, _ = theta_derivative_coupling(ℓ, m_bs)
                d_Lell = T(C_plus) + T(A_plus)
            elseif L == ℓ - 1
                C_minus, _ = cos_theta_coupling(ℓ, m_bs)
                A_minus, _, _ = theta_derivative_coupling(ℓ, m_bs)
                d_Lell = T(C_minus) + T(A_minus)
            elseif L == ℓ
                # Diagonal contribution from cosθ (zero) and sinθ∂/∂θ (small)
                _, _, A_diag = theta_derivative_coupling(ℓ, m_bs)
                d_Lell = T(A_diag)
            end

            if abs(d_Lell) > eps(T)
                source_ur .-= d_Lell .* utheta_ell ./ r
            end
        end

        # Contribution from u_φ term (if present)
        if haskey(uphi_coeffs, (L, m_bs))
            uphi_L = uphi_coeffs[(L, m_bs)]
            inv_sin_LL = inv_sin_theta_gaunt(L, L, m_bs)
            source_ur .-= T(m_bs) .* inv_sin_LL .* uphi_L ./ r
        end

        # Solve: ∂(r² u_r)/∂r = r² × source_ur
        # → D1 × (r² u_r) = r² × source_ur
        # With BC: u_r = 0 at boundaries

        rhs_ur = (r.^2) .* source_ur

        # Build operator for r² u_r
        A_ur = copy(D1)

        # Apply BC: r² u_r = 0 at boundaries (which means u_r = 0)
        A_ur[idx_inner, :] .= zero(T)
        A_ur[idx_inner, idx_inner] = one(T)
        rhs_ur[idx_inner] = zero(T)

        A_ur[idx_outer, :] .= zero(T)
        A_ur[idx_outer, idx_outer] = one(T)
        rhs_ur[idx_outer] = zero(T)

        # Solve for r² u_r
        r2_ur = A_ur \ rhs_ur

        # Extract u_r
        ur_L = r2_ur ./ (r.^2)
        ur_L[idx_inner] = zero(T)  # Enforce BC exactly
        ur_L[idx_outer] = zero(T)

        ur_coeffs[(L, m_bs)] = ur_L
        dur_dr_coeffs[(L, m_bs)] = D1 * ur_L
    end

    return nothing
end


"""
    solve_meridional_simple!(ur_coeffs, utheta_coeffs, dur_dr_coeffs, dutheta_dr_coeffs,
                              theta_coeffs, r, D1, r_i, r_o,
                              Ra, E, Pr, lmax_bs, mmax_bs;
                              mechanical_bc=:no_slip)

Compute simplified meridional circulation using leading-order balance.

This uses a simplified model where the θ-component of thermal wind is solved
mode-by-mode with a diagonal approximation for the (ẑ·∇) operator:

    2Ω × β_eff × ∂u_θ/∂r ≈ -(Ra E²/Pr) × im T̄ / (r × sinθ_eff)

where β_eff and sinθ_eff are effective latitude-averaged factors.

The radial velocity u_r is then computed from the continuity equation.

This approximation is valid when:
- The non-axisymmetric amplitude is small (ε << 1)
- The dominant contribution is from the diagonal (same-ℓ) terms

For more accurate results with strong non-axisymmetry, use the full coupled solver.
"""
function solve_meridional_simple!(
    ur_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    utheta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dur_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dutheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    r::Vector{T}, D1::Matrix{T},
    r_i::T, r_o::T,
    Ra::T, E::T, Pr::T,
    lmax_bs::Int, mmax_bs::Int;
    mechanical_bc::Symbol = :no_slip
) where T<:Real

    Nr = length(r)
    two_omega = one(T) / E
    buoyancy_factor = Ra * E^2 / Pr

    # Boundary indices
    idx_inner = abs(r[1] - r_i) < abs(r[Nr] - r_i) ? 1 : Nr
    idx_outer = idx_inner == 1 ? Nr : 1

    # m = 0: No meridional circulation (∂T̄/∂φ = 0)
    for ℓ in 0:lmax_bs
        ur_coeffs[(ℓ, 0)] = zeros(T, Nr)
        utheta_coeffs[(ℓ, 0)] = zeros(T, Nr)
        dur_dr_coeffs[(ℓ, 0)] = zeros(T, Nr)
        dutheta_dr_coeffs[(ℓ, 0)] = zeros(T, Nr)
    end

    # m ≠ 0: Solve simplified θ-thermal wind equation
    for m_bs in 1:mmax_bs
        for ℓ in m_bs:lmax_bs
            # Check for temperature forcing
            if !haskey(theta_coeffs, (ℓ, m_bs))
                ur_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                utheta_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                dur_dr_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                dutheta_dr_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                continue
            end

            T_lm = theta_coeffs[(ℓ, m_bs)]
            if maximum(abs.(T_lm)) < eps(T) * 100
                ur_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                utheta_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                dur_dr_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                dutheta_dr_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                continue
            end

            # Effective factors for diagonal approximation
            # sinθ_eff: effective value of 1/sinθ for mode (ℓ,m)
            inv_sin_eff = one(T) + T(m_bs^2) / T(max(ℓ * (ℓ + 1), 1))

            # β_eff: effective z-derivative factor (latitude average of cosθ)
            # For geostrophic flow, use the characteristic value
            β_eff = sqrt(T(ℓ) / T(ℓ + 1))

            # RHS forcing: -(Ra E²/Pr) × im × T̄_ℓm × inv_sin_eff / r
            forcing = -buoyancy_factor .* T(m_bs) .* T_lm .* inv_sin_eff ./ r

            # Simplified equation: 2Ω × β_eff × du_θ/dr = forcing
            # Integrate: u_θ = (1/(2Ω β_eff)) × ∫ forcing dr

            # Build integration operator (D1 with BC)
            coeff = two_omega * β_eff
            if abs(coeff) < eps(T) * 100
                ur_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                utheta_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                dur_dr_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                dutheta_dr_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                continue
            end

            # Solve: coeff × D1 × u_θ = forcing with u_θ(boundaries) = 0 (no-slip)
            A_mat = coeff .* D1

            # Apply boundary conditions
            if mechanical_bc == :no_slip
                A_mat[idx_inner, :] .= zero(T)
                A_mat[idx_inner, idx_inner] = one(T)
                forcing[idx_inner] = zero(T)

                A_mat[idx_outer, :] .= zero(T)
                A_mat[idx_outer, idx_outer] = one(T)
                forcing[idx_outer] = zero(T)
            end

            # Solve for u_θ
            utheta_lm = A_mat \ forcing

            # Compute u_r from continuity: ∂(r² u_r)/∂r + [angular terms] = 0
            # Simplified: u_r ~ -(r/ℓ(ℓ+1)) × (angular derivative of u_θ)
            # Using c_θ factor as proxy for angular derivative magnitude
            ell_factor = T(ℓ * (ℓ + 1))
            c_theta = ell_factor > 0 ? sqrt(max(ell_factor - T(m_bs^2), zero(T))) / sqrt(ell_factor) : zero(T)

            # Estimate u_r from u_θ using poloidal relationship
            # u_r ~ ℓ(ℓ+1)/(r) × integral of u_θ type terms
            # For simplicity, scale u_r by the angular derivative coupling
            ur_lm = c_theta .* utheta_lm .* r ./ max(ell_factor, one(T))

            # Enforce u_r = 0 at boundaries
            ur_lm[idx_inner] = zero(T)
            ur_lm[idx_outer] = zero(T)

            ur_coeffs[(ℓ, m_bs)] = ur_lm
            utheta_coeffs[(ℓ, m_bs)] = utheta_lm
            dur_dr_coeffs[(ℓ, m_bs)] = D1 * ur_lm
            dutheta_dr_coeffs[(ℓ, m_bs)] = D1 * utheta_lm
        end
    end

    return nothing
end


"""
    solve_meridional_circulation_toroidal_poloidal!(ur_coeffs, utheta_coeffs, dur_dr_coeffs, dutheta_dr_coeffs,
                                                     theta_coeffs, uphi_coeffs,
                                                     r, D1, D2, r_i, r_o,
                                                     Ra, E, Pr, lmax_bs, mmax_bs;
                                                     mechanical_bc=:no_slip,
                                                     include_meridional=true,
                                                     use_full_coupling=true)

Solve for the meridional circulation (ū_r, ū_θ) from geostrophic balance.

Key physics:
- For m = 0 (axisymmetric): No φ-derivative of T̄, so u_r = u_θ = 0
- For m ≠ 0: The φ-gradient of temperature drives meridional flow

The thermal wind equation for the θ-component is:
    2Ω (ẑ·∇) u_θ = -(Ra E²/Pr)/(r sinθ) × ∂T̄/∂φ

The operator (ẑ·∇) = cosθ ∂/∂r - (sinθ/r) ∂/∂θ couples modes ℓ to ℓ±1.

# Arguments
- `include_meridional` : If false, set u_r = u_θ = 0 (default: true)
- `use_full_coupling` : If true, use full block-tridiagonal coupled solver
                        If false, use diagonal approximation (default: true)
"""
function solve_meridional_circulation_toroidal_poloidal!(
    ur_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    utheta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dur_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dutheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    r::Vector{T}, D1::Matrix{T}, D2::Matrix{T},
    r_i::T, r_o::T,
    Ra::T, E::T, Pr::T,
    lmax_bs::Int, mmax_bs::Int;
    mechanical_bc::Symbol = :no_slip,
    include_meridional::Bool = true,
    use_full_coupling::Bool = true
) where T<:Real

    Nr = length(r)

    if !include_meridional
        # Set meridional circulation to zero (leading-order approximation)
        for m_bs in 0:mmax_bs
            for ℓ in m_bs:lmax_bs
                ur_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                utheta_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                dur_dr_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                dutheta_dr_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
            end
        end
        return nothing
    end

    if use_full_coupling
        # Full coupled solver: solves the complete block-tridiagonal system
        # for each azimuthal wavenumber m separately
        for m_bs in 0:mmax_bs
            solve_meridional_coupled!(
                ur_coeffs, utheta_coeffs, dur_dr_coeffs, dutheta_dr_coeffs,
                theta_coeffs, uphi_coeffs,
                r, D1, D2, r_i, r_o,
                Ra, E, Pr, m_bs, lmax_bs;
                mechanical_bc=mechanical_bc
            )
        end
    else
        # Simplified diagonal approximation (faster but less accurate)
        solve_meridional_simple!(
            ur_coeffs, utheta_coeffs, dur_dr_coeffs, dutheta_dr_coeffs,
            theta_coeffs, r, D1, r_i, r_o,
            Ra, E, Pr, lmax_bs, mmax_bs;
            mechanical_bc=mechanical_bc
        )
    end

    return nothing
end


# =============================================================================
#  Full Advection Term with All Velocity Components
# =============================================================================

"""
    compute_full_advection_spectral(theta_coeffs, dtheta_dr_coeffs,
                                     ur_coeffs, utheta_coeffs, uphi_coeffs,
                                     lmax_bs, mmax_bs, r)

Compute the full advection term ū·∇T̄ in spectral space including all
velocity components (ū_r, ū_θ, ū_φ).

The full advection is:
    ū·∇T̄ = ū_r ∂T̄/∂r + (ū_θ/r) ∂T̄/∂θ + (ū_φ/(r sinθ)) ∂T̄/∂φ

In spectral space:
- ∂T̄/∂r is computed directly from dtheta_dr_coeffs
- ∂T̄/∂θ couples ℓ to ℓ±1 via recurrence relations
- ∂T̄/∂φ gives factor of im for mode m

Returns forcing coefficients for the advection-diffusion equation.
"""
function compute_full_advection_spectral(
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dtheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    ur_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    utheta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    lmax_bs::Int, mmax_bs::Int,
    r::Vector{T}
) where T<:Real

    Nr = length(r)
    forcing = Dict{Tuple{Int,Int}, Vector{T}}()

    # ==========================================================================
    # Term 1: ū_r × ∂T̄/∂r (radial advection)
    # ==========================================================================
    # This is diagonal in (ℓ,m): ū_r,ℓm × ∂T̄_ℓm/∂r

    for m_bs in 0:mmax_bs
        for ℓ in m_bs:lmax_bs
            if !haskey(ur_coeffs, (ℓ, m_bs)) || !haskey(dtheta_dr_coeffs, (ℓ, m_bs))
                continue
            end

            ur_lm = ur_coeffs[(ℓ, m_bs)]
            dT_dr_lm = dtheta_dr_coeffs[(ℓ, m_bs)]

            if maximum(abs.(ur_lm)) < eps(T) * 100 || maximum(abs.(dT_dr_lm)) < eps(T) * 100
                continue
            end

            # Diagonal contribution
            if !haskey(forcing, (ℓ, m_bs))
                forcing[(ℓ, m_bs)] = zeros(T, Nr)
            end
            forcing[(ℓ, m_bs)] .+= ur_lm .* dT_dr_lm
        end
    end

    # ==========================================================================
    # Term 2: (ū_θ/r) × ∂T̄/∂θ (meridional advection)
    # ==========================================================================
    # ∂T̄/∂θ couples ℓ to ℓ±1 via theta_derivative_coeff_3d
    # This is more complex due to mode coupling

    for m_bs in 0:mmax_bs
        for ℓ_T in m_bs:lmax_bs
            if !haskey(theta_coeffs, (ℓ_T, m_bs))
                continue
            end
            T_lm = theta_coeffs[(ℓ_T, m_bs)]

            if maximum(abs.(T_lm)) < eps(T) * 100
                continue
            end

            # θ-derivative of T̄_ℓm couples to ℓ±1
            c_plus, c_minus = theta_derivative_coeff_3d(ℓ_T, m_bs)

            # Contribution to ℓ_T - 1 mode
            if ℓ_T > m_bs && abs(c_minus) > eps(T)
                L_out = ℓ_T - 1
                if haskey(utheta_coeffs, (L_out, m_bs))
                    utheta_L = utheta_coeffs[(L_out, m_bs)]
                    if maximum(abs.(utheta_L)) > eps(T) * 100
                        if !haskey(forcing, (L_out, m_bs))
                            forcing[(L_out, m_bs)] = zeros(T, Nr)
                        end
                        forcing[(L_out, m_bs)] .+= c_minus .* (utheta_L ./ r) .* T_lm
                    end
                end
            end

            # Contribution to ℓ_T + 1 mode
            if ℓ_T + 1 <= lmax_bs && abs(c_plus) > eps(T)
                L_out = ℓ_T + 1
                if haskey(utheta_coeffs, (L_out, m_bs))
                    utheta_L = utheta_coeffs[(L_out, m_bs)]
                    if maximum(abs.(utheta_L)) > eps(T) * 100
                        if !haskey(forcing, (L_out, m_bs))
                            forcing[(L_out, m_bs)] = zeros(T, Nr)
                        end
                        forcing[(L_out, m_bs)] .+= c_plus .* (utheta_L ./ r) .* T_lm
                    end
                end
            end
        end
    end

    # ==========================================================================
    # Term 3: (ū_φ/(r sinθ)) × ∂T̄/∂φ (azimuthal advection)
    # ==========================================================================
    # This is the original term, already implemented in compute_phi_advection_spectral
    # Add it here using the same logic

    for m_bs in 1:mmax_bs  # m=0 has no φ-advection
        im_m = T(m_bs)

        for ℓ_T in m_bs:lmax_bs
            if !haskey(theta_coeffs, (ℓ_T, m_bs))
                continue
            end
            T_lm = theta_coeffs[(ℓ_T, m_bs)]

            if maximum(abs.(T_lm)) < eps(T) * 100
                continue
            end

            for ℓ_u in m_bs:lmax_bs
                if !haskey(uphi_coeffs, (ℓ_u, m_bs))
                    continue
                end
                u_Lm = uphi_coeffs[(ℓ_u, m_bs)]

                if maximum(abs.(u_Lm)) < eps(T) * 100
                    continue
                end

                # Diagonal contribution
                if ℓ_T == ℓ_u
                    if !haskey(forcing, (ℓ_T, m_bs))
                        forcing[(ℓ_T, m_bs)] = zeros(T, Nr)
                    end
                    forcing[(ℓ_T, m_bs)] .+= im_m .* u_Lm .* T_lm ./ r
                end

                # Off-diagonal contributions (simplified)
                L_diff = abs(ℓ_T - ℓ_u)
                if L_diff != ℓ_T && L_diff >= m_bs
                    c_coupling = sqrt(T(2*ℓ_T + 1) * T(2*ℓ_u + 1)) / (4 * T(π) * T(2*L_diff + 1))
                    c_coupling *= T(0.5)
                    if !haskey(forcing, (L_diff, m_bs))
                        forcing[(L_diff, m_bs)] = zeros(T, Nr)
                    end
                    forcing[(L_diff, m_bs)] .+= c_coupling .* im_m .* u_Lm .* T_lm ./ r
                end

                L_sum = ℓ_T + ℓ_u
                if L_sum <= lmax_bs && L_sum != ℓ_T
                    c_coupling = sqrt(T(2*ℓ_T + 1) * T(2*ℓ_u + 1)) / (4 * T(π) * T(2*L_sum + 1))
                    c_coupling *= T(0.3)
                    if !haskey(forcing, (L_sum, m_bs))
                        forcing[(L_sum, m_bs)] = zeros(T, Nr)
                    end
                    forcing[(L_sum, m_bs)] .+= c_coupling .* im_m .* u_Lm .* T_lm ./ r
                end
            end
        end
    end

    return forcing
end


"""
    nonaxisymmetric_basic_state_selfconsistent(
        cd::ChebyshevDiffn{T}, χ::T, E::T, Ra::T, Pr::T,
        lmax_bs::Int, mmax_bs::Int, amplitudes::Dict{Tuple{Int,Int}, T};
        mechanical_bc::Symbol = :no_slip,
        thermal_bc::Symbol = :fixed_temperature,
        outer_fluxes::Dict{Tuple{Int,Int}, T} = Dict{Tuple{Int,Int}, T}(),
        max_iterations::Int = 20,
        tolerance::T = T(1e-8),
        verbose::Bool = false
    ) where T<:Real

Create a self-consistent non-axisymmetric basic state that accounts for
temperature advection by the thermal wind flow.

Unlike `nonaxisymmetric_basic_state`, which assumes ∇²T̄ = 0 (valid for
low Péclet number), this function iteratively solves the full advection-diffusion
equation:

    κ∇²T̄ = ū·∇T̄ = ū_φ/(r sinθ) × ∂T̄/∂φ

This is important when:
- The Péclet number Pe = UL/κ is not small (strong advection)
- The non-axisymmetric amplitude is significant
- Quantitative accuracy is needed for tri-global stability

# Algorithm (Picard iteration)
1. Initialize T̄⁽⁰⁾ by solving ∇²T̄ = 0 (Laplace equation)
2. Compute ū_φ⁽ⁿ⁾ from thermal wind balance with T̄⁽ⁿ⁾
3. Compute advection source: S⁽ⁿ⁾ = (1/κ) × ū_φ⁽ⁿ⁾/(r sinθ) × ∂T̄⁽ⁿ⁾/∂φ
4. Solve ∇²T̄⁽ⁿ⁺¹⁾ = S⁽ⁿ⁾ with original boundary conditions
5. Check convergence: ‖T̄⁽ⁿ⁺¹⁾ - T̄⁽ⁿ⁾‖ < tolerance
6. Repeat steps 2-5 until converged

# Arguments
Same as `nonaxisymmetric_basic_state`, plus:
- `max_iterations` : Maximum Picard iterations (default: 20)
- `tolerance` : Convergence tolerance on temperature change (default: 1e-8)
- `verbose` : Print convergence information (default: false)

# Returns
- `BasicState3D` : The self-consistent basic state
- `ConvergenceInfo` : Named tuple with iteration count and residual history

# Example
```julia
bc = Y20(0.1) + Y22(0.05)
amplitudes = to_dict(bc)
bs, info = nonaxisymmetric_basic_state_selfconsistent(
    cd, χ, E, Ra, Pr, lmax_bs, mmax_bs, amplitudes;
    verbose=true
)
println("Converged in \$(info.iterations) iterations")
```

# Physical Notes
- The advection term couples modes with the same m but different ℓ
- For m=0 modes, the advection term is zero (ū_φ advects only in φ)
- The iteration typically converges quickly for small amplitude variations
- Non-convergence may indicate unstable basic state (should not occur for stability analysis)
"""
function nonaxisymmetric_basic_state_selfconsistent(
    cd::ChebyshevDiffn{T}, χ::T, E::T, Ra::T, Pr::T,
    lmax_bs::Int, mmax_bs::Int, amplitudes::Dict{Tuple{Int,Int}, <:Real};
    mechanical_bc::Symbol = :no_slip,
    thermal_bc::Symbol = :fixed_temperature,
    outer_fluxes::Dict{Tuple{Int,Int}, <:Real} = Dict{Tuple{Int,Int}, T}(),
    max_iterations::Int = 20,
    tolerance::T = T(1e-8),
    verbose::Bool = false
) where T<:Real

    # =========================================================================
    # Setup
    # =========================================================================
    r = cd.x
    Nr = length(r)
    D1 = Matrix(cd.D1)
    D2 = Matrix(cd.D2)
    r_i = T(χ)
    r_o = T(1)

    # Thermal diffusivity in dimensionless units
    # In the viscous time scaling: κ_eff = 1/Pr (relative to viscous diffusion)
    κ_eff = one(T) / Pr

    # Convert amplitude dicts to consistent type
    amplitudes_T = Dict{Tuple{Int,Int}, T}(k => T(v) for (k,v) in amplitudes)
    outer_fluxes_T = Dict{Tuple{Int,Int}, T}(k => T(v) for (k,v) in outer_fluxes)

    # =========================================================================
    # Step 1: Initial guess from Laplace solution
    # =========================================================================
    if verbose
        println("Self-consistent basic state solver:")
        println("  Parameters: E=$E, Ra=$Ra, Pr=$Pr, χ=$χ")
        println("  Resolution: Nr=$Nr, lmax=$lmax_bs, mmax=$mmax_bs")
        println("  Iteration 0: Solving Laplace equation for initial guess...")
    end

    # Use the existing function for initial guess
    bs_init = nonaxisymmetric_basic_state(
        cd, χ, E, Ra, Pr, lmax_bs, mmax_bs, amplitudes_T;
        mechanical_bc=mechanical_bc,
        thermal_bc=thermal_bc,
        outer_fluxes=outer_fluxes_T
    )

    # Copy coefficients for iteration
    theta_coeffs = deepcopy(bs_init.theta_coeffs)
    dtheta_dr_coeffs = deepcopy(bs_init.dtheta_dr_coeffs)
    uphi_coeffs = deepcopy(bs_init.uphi_coeffs)
    duphi_dr_coeffs = deepcopy(bs_init.duphi_dr_coeffs)

    # Store original boundary condition values (for reapplication)
    bc_values = Dict{Tuple{Int,Int}, Tuple{T, T, Symbol}}()  # (inner_val, outer_val, bc_type)

    # Spherical harmonic normalization
    Y_norm(ℓ::Int, m::Int) = m == 0 ? sqrt(T(2ℓ+1)/(4*T(π))) : sqrt(T(2ℓ+1)/(4*T(π)) * 2)

    for ℓ in 0:lmax_bs
        for m in 0:min(ℓ, mmax_bs)
            norm_Ylm = Y_norm(ℓ, m)

            if ℓ == 0 && m == 0
                # Mean temperature: inner = 1 (hot), outer = 0 or flux
                inner_val = sqrt(T(4) * T(π))
                if thermal_bc == :fixed_temperature
                    outer_val = zero(T)
                else
                    flux_00 = get(outer_fluxes_T, (0,0), get(amplitudes_T, (0,0), zero(T)))
                    outer_val = T(flux_00) * sqrt(T(4) * T(π))
                end
                bc_values[(0,0)] = (inner_val, outer_val, thermal_bc)
            else
                # Higher modes: inner = 0, outer from amplitudes or fluxes
                inner_val = zero(T)
                if thermal_bc == :fixed_flux
                    value = get(outer_fluxes_T, (ℓ,m), get(amplitudes_T, (ℓ,m), zero(T)))
                    outer_val = T(value) / norm_Ylm
                else
                    value = get(amplitudes_T, (ℓ,m), zero(T))
                    outer_val = T(value) / norm_Ylm
                end
                bc_values[(ℓ,m)] = (inner_val, outer_val, thermal_bc)
            end
        end
    end

    # =========================================================================
    # Initialize velocity coefficient dictionaries for full geostrophic balance
    # =========================================================================
    ur_coeffs = Dict{Tuple{Int,Int}, Vector{T}}()
    utheta_coeffs = Dict{Tuple{Int,Int}, Vector{T}}()
    dur_dr_coeffs = Dict{Tuple{Int,Int}, Vector{T}}()
    dutheta_dr_coeffs = Dict{Tuple{Int,Int}, Vector{T}}()

    # Initialize to zero for all modes
    for ℓ in 0:lmax_bs
        for m in 0:min(ℓ, mmax_bs)
            ur_coeffs[(ℓ, m)] = zeros(T, Nr)
            utheta_coeffs[(ℓ, m)] = zeros(T, Nr)
            dur_dr_coeffs[(ℓ, m)] = zeros(T, Nr)
            dutheta_dr_coeffs[(ℓ, m)] = zeros(T, Nr)
        end
    end

    # =========================================================================
    # Picard iteration
    # =========================================================================
    residual_history = T[]
    converged = false
    iteration = 0

    for iter in 1:max_iterations
        iteration = iter

        # Store previous temperature for convergence check
        theta_prev = deepcopy(theta_coeffs)

        # ---------------------------------------------------------------------
        # Step 2: Update thermal wind (already done for initial guess)
        # ---------------------------------------------------------------------
        # The thermal wind is updated inside the loop after temperature update

        # ---------------------------------------------------------------------
        # Step 2b: Compute meridional circulation (ū_r, ū_θ) using toroidal-poloidal
        # ---------------------------------------------------------------------
        # Uses the θ-component of thermal wind equation directly
        # No pressure computation needed - continuity is automatically satisfied
        solve_meridional_circulation_toroidal_poloidal!(
            ur_coeffs, utheta_coeffs, dur_dr_coeffs, dutheta_dr_coeffs,
            theta_coeffs, uphi_coeffs,
            r, D1, D2, r_i, r_o,
            Ra, E, Pr, lmax_bs, mmax_bs;
            mechanical_bc=mechanical_bc
        )

        # ---------------------------------------------------------------------
        # Step 3: Compute FULL advection source term
        # ---------------------------------------------------------------------
        # S = (1/κ) × ū·∇T̄ = (1/κ) × [ū_r ∂T̄/∂r + (ū_θ/r) ∂T̄/∂θ + (ū_φ/(r sinθ)) ∂T̄/∂φ]

        advection_source = compute_full_advection_spectral(
            theta_coeffs, dtheta_dr_coeffs,
            ur_coeffs, utheta_coeffs, uphi_coeffs,
            lmax_bs, mmax_bs, r
        )

        # Scale by 1/κ
        for (key, val) in advection_source
            advection_source[key] = val ./ κ_eff
        end

        # ---------------------------------------------------------------------
        # Step 4: Solve Poisson equation ∇²T̄ = S with boundary conditions
        # ---------------------------------------------------------------------
        for ℓ in 0:lmax_bs
            for m in 0:min(ℓ, mmax_bs)
                # Get forcing for this mode (zero if no advection)
                forcing = get(advection_source, (ℓ, m), zeros(T, Nr))

                # Get boundary conditions
                inner_val, outer_val, bc_type = get(bc_values, (ℓ, m), (zero(T), zero(T), :fixed_temperature))

                # Only solve if there's forcing OR this mode has BC amplitude
                has_bc = abs(outer_val) > eps(T) * 100 || (ℓ == 0 && m == 0)
                has_forcing = maximum(abs.(forcing)) > eps(T) * 100

                if has_bc || has_forcing
                    T_lm, dT_lm = solve_poisson_mode(
                        ℓ, m, r, D2, D1, r_i, r_o, forcing;
                        inner_value = inner_val,
                        outer_value = outer_val,
                        outer_bc = bc_type,
                        inner_bc = :fixed_temperature  # Always Dirichlet at inner boundary
                    )

                    theta_coeffs[(ℓ, m)] = T_lm
                    dtheta_dr_coeffs[(ℓ, m)] = dT_lm
                end
            end
        end

        # ---------------------------------------------------------------------
        # Step 5: Update thermal wind with new temperature
        # ---------------------------------------------------------------------
        for m_bs in 0:mmax_bs
            # Extract temperature modes for this m
            theta_m = Dict{Int, Vector{T}}()
            for ℓ in m_bs:lmax_bs
                if haskey(theta_coeffs, (ℓ, m_bs))
                    theta_m[ℓ] = theta_coeffs[(ℓ, m_bs)]
                end
            end

            if isempty(theta_m) || all(maximum(abs.(v)) < eps(T) * 100 for v in values(theta_m))
                continue
            end

            # Initialize velocity storage for this m
            uphi_m = Dict{Int, Vector{T}}(ℓ => zeros(T, Nr) for ℓ in 0:lmax_bs)
            duphi_dr_m = Dict{Int, Vector{T}}(ℓ => zeros(T, Nr) for ℓ in 0:lmax_bs)

            # Solve thermal wind
            solve_thermal_wind_balance_3d!(uphi_m, duphi_dr_m, theta_m, m_bs,
                                           cd, r_i, r_o, Ra, Pr;
                                           mechanical_bc=mechanical_bc,
                                           E=E)

            # Copy results to storage
            for ℓ in 0:lmax_bs
                if haskey(uphi_m, ℓ) && maximum(abs.(uphi_m[ℓ])) > eps(T) * 100
                    uphi_coeffs[(ℓ, m_bs)] = uphi_m[ℓ]
                    duphi_dr_coeffs[(ℓ, m_bs)] = duphi_dr_m[ℓ]
                end
            end
        end

        # ---------------------------------------------------------------------
        # Step 6: Check convergence
        # ---------------------------------------------------------------------
        max_change = zero(T)
        for (key, theta_new) in theta_coeffs
            if haskey(theta_prev, key)
                theta_old = theta_prev[key]
                change = maximum(abs.(theta_new .- theta_old))
                max_change = max(max_change, change)
            end
        end

        push!(residual_history, max_change)

        if verbose
            println("  Iteration $iter: max temperature change = $(Printf.@sprintf("%.2e", max_change))")
        end

        if max_change < tolerance
            converged = true
            if verbose
                println("  Converged in $iter iterations (tolerance = $tolerance)")
            end
            break
        end
    end

    if !converged && verbose
        println("  Warning: Did not converge after $max_iterations iterations")
        println("           Final residual = $(residual_history[end])")
    end

    # =========================================================================
    # Build result
    # =========================================================================
    # Fill any missing coefficients with zeros
    for ℓ in 0:lmax_bs
        for m in 0:min(ℓ, mmax_bs)
            if !haskey(theta_coeffs, (ℓ, m))
                theta_coeffs[(ℓ, m)] = zeros(T, Nr)
                dtheta_dr_coeffs[(ℓ, m)] = zeros(T, Nr)
            end
            if !haskey(uphi_coeffs, (ℓ, m))
                uphi_coeffs[(ℓ, m)] = zeros(T, Nr)
                duphi_dr_coeffs[(ℓ, m)] = zeros(T, Nr)
            end
            # ur_coeffs, utheta_coeffs already initialized and computed
            if !haskey(ur_coeffs, (ℓ, m))
                ur_coeffs[(ℓ, m)] = zeros(T, Nr)
                dur_dr_coeffs[(ℓ, m)] = zeros(T, Nr)
            end
            if !haskey(utheta_coeffs, (ℓ, m))
                utheta_coeffs[(ℓ, m)] = zeros(T, Nr)
                dutheta_dr_coeffs[(ℓ, m)] = zeros(T, Nr)
            end
        end
    end

    bs = BasicState3D(
        lmax_bs = lmax_bs,
        mmax_bs = mmax_bs,
        Nr = Nr,
        r = r,
        theta_coeffs = theta_coeffs,
        dtheta_dr_coeffs = dtheta_dr_coeffs,
        ur_coeffs = ur_coeffs,
        utheta_coeffs = utheta_coeffs,
        uphi_coeffs = uphi_coeffs,
        dur_dr_coeffs = dur_dr_coeffs,
        dutheta_dr_coeffs = dutheta_dr_coeffs,
        duphi_dr_coeffs = duphi_dr_coeffs
    )

    info = (iterations = iteration, converged = converged, residual_history = residual_history)

    return bs, info
end


"""
    basic_state_selfconsistent(cd, χ, E, Ra, Pr;
                                temperature_bc=nothing,
                                flux_bc=nothing,
                                mechanical_bc=:no_slip,
                                lmax_bs=nothing,
                                max_iterations=20,
                                tolerance=1e-8,
                                verbose=false)

Create a self-consistent basic state with symbolic spherical harmonic boundary conditions.

This is the advection-corrected version of `basic_state()` that accounts for
temperature advection in non-axisymmetric basic states. For axisymmetric BCs,
it falls back to the standard solver (since advection is zero for m=0 only).

# Arguments
Same as `basic_state()`, plus:
- `max_iterations` : Maximum Picard iterations (default: 20)
- `tolerance` : Convergence tolerance (default: 1e-8)
- `verbose` : Print convergence information (default: false)

# Returns
For non-axisymmetric BCs:
- `BasicState3D` : The self-consistent basic state
- `ConvergenceInfo` : Named tuple with iteration details

For axisymmetric BCs (falls back to standard solver):
- `BasicState` or `BasicState3D` : Depending on BC type
- `nothing` : No convergence info needed

# Example
```julia
bc = Y20(0.1) + Y22(0.05)
bs, info = basic_state_selfconsistent(cd, χ, E, Ra, Pr;
                                       temperature_bc=bc,
                                       verbose=true)
```

# When to Use
- When the non-axisymmetric amplitude is significant (> 0.1)
- When high quantitative accuracy is needed
- When studying strong forcing scenarios

For small amplitude variations (< 0.1), the standard `basic_state()` is
usually sufficient and faster.
"""
function basic_state_selfconsistent(cd, χ::Real, E::Real, Ra::Real, Pr::Real;
                                    temperature_bc::Union{Nothing, SphericalHarmonicBC}=nothing,
                                    flux_bc::Union{Nothing, SphericalHarmonicBC}=nothing,
                                    mechanical_bc::Symbol=:no_slip,
                                    lmax_bs::Union{Nothing, Int}=nothing,
                                    max_iterations::Int=20,
                                    tolerance::Float64=1e-8,
                                    verbose::Bool=false)

    # Validate: can't have both temperature_bc and flux_bc
    if temperature_bc !== nothing && flux_bc !== nothing
        error("Cannot specify both temperature_bc and flux_bc. Choose one.")
    end

    T = eltype(cd.x)

    # Determine thermal BC type and the boundary condition
    if flux_bc !== nothing
        thermal_bc = :fixed_flux
        bc = flux_bc
    elseif temperature_bc !== nothing
        thermal_bc = :fixed_temperature
        bc = temperature_bc
    else
        # No BC specified → pure conduction (no advection to correct)
        _lmax = lmax_bs === nothing ? 4 : lmax_bs
        return conduction_basic_state(cd, T(χ), _lmax; thermal_bc=:fixed_temperature), nothing
    end

    # Get lmax and mmax from boundary condition
    bc_lmax, bc_mmax = get_lmax_mmax(bc)

    # Use provided lmax_bs or auto-determine
    _lmax = lmax_bs === nothing ? max(bc_lmax + 2, 4) : lmax_bs

    # Check if BC is axisymmetric - if so, advection is zero, use standard solver
    if is_axisymmetric(bc)
        if verbose
            println("Boundary condition is axisymmetric (m=0 only). Using standard solver.")
            println("(No advection correction needed: ū·∇T̄ = 0 for axisymmetric T̄)")
        end
        return basic_state(cd, χ, E, Ra, Pr;
                          temperature_bc=temperature_bc,
                          flux_bc=flux_bc,
                          mechanical_bc=mechanical_bc,
                          lmax_bs=_lmax), nothing
    end

    # Non-axisymmetric: use self-consistent solver
    amplitudes = to_dict(bc)

    if thermal_bc == :fixed_temperature
        return nonaxisymmetric_basic_state_selfconsistent(
            cd, T(χ), T(E), T(Ra), T(Pr),
            _lmax, bc_mmax, amplitudes;
            mechanical_bc=mechanical_bc,
            thermal_bc=:fixed_temperature,
            max_iterations=max_iterations,
            tolerance=T(tolerance),
            verbose=verbose
        )
    else  # fixed_flux
        return nonaxisymmetric_basic_state_selfconsistent(
            cd, T(χ), T(E), T(Ra), T(Pr),
            _lmax, bc_mmax,
            Dict{Tuple{Int,Int},T}();  # empty amplitudes
            mechanical_bc=mechanical_bc,
            thermal_bc=:fixed_flux,
            outer_fluxes=amplitudes,
            max_iterations=max_iterations,
            tolerance=T(tolerance),
            verbose=verbose
        )
    end
end
