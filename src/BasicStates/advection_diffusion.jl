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

_value_real_type(::Type{T}) where {T<:Real} = T
_value_real_type(::Type{Complex{T}}) where {T<:Real} = T

@inline _maxabs(v) = maximum(abs, v)

function _maxabsdiff(a, b)
    R = promote_type(_value_real_type(eltype(a)), _value_real_type(eltype(b)))
    out = zero(R)
    @inbounds for i in eachindex(a, b)
        diff = abs(a[i] - b[i])
        out = max(out, R(diff))
    end
    return out
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

    # Azimuthal advection ū_φ·∂_φT̄ projects to ZERO in the real-orthonormal
    # cos(mφ) basis the basic state is stored in: ∂_φ maps cos(mφ)→sin(mφ), so
    # ū_φ·∂_φT̄ ~ cos·sin = pure sin, orthogonal to every cos(mφ) basis function.
    # (Verified to machine precision by a manufactured-solution test.)
    #
    # The previous implementation used arbitrary "empirical reduction factors"
    # (0.5, 0.3) and ∂_φ → ×m, producing spurious nonzero forcing — removed.
    #
    # The genuine φ-advection is now captured by `vecsh_advection` (divergence form
    # on the full ±m real-SH basis), which routes the cos→sin product into the
    # `sin(mφ)` (`-m`) coefficients. This standalone scalar projection remains zero
    # by construction and is retained only as a documented building block.
    return Dict{Tuple{Int,Int}, Vector{T}}()
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

    # Build the dense radial Laplacian directly.  The equivalent Diagonal-based
    # expression materializes several Nr-by-Nr temporaries inside the modal loop.
    A_mat = copy(D2)
    ℓ_factor = T(ℓ * (ℓ + 1))
    @inbounds for i in 1:Nr
        inv_r = inv(r[i])
        d1_scale = T(2) * inv_r
        for j in 1:Nr
            A_mat[i, j] += d1_scale * D1[i, j]
        end
        A_mat[i, i] -= ℓ_factor * inv_r * inv_r
    end

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
        @inbounds for j in 1:Nr
            A_mat[idx_inner, j] = D1[idx_inner, j]
        end
        f_rhs[idx_inner] = inner_value  # This is the flux value
    end

    # Outer boundary condition
    if outer_bc == :fixed_temperature
        A_mat[idx_outer, :] .= zero(T)
        A_mat[idx_outer, idx_outer] = one(T)
        f_rhs[idx_outer] = outer_value
    else  # fixed_flux
        @inbounds for j in 1:Nr
            A_mat[idx_outer, j] = D1[idx_outer, j]
        end
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
    # Exact, finite two-term identity (verified numerically against the
    # orthonormal Y_lm; see basic_state.jl:2007-2008 and _dtheta_sphere_projection):
    #     sinθ ∂Y_ℓm/∂θ = ℓ α⁺_ℓ Y_{ℓ+1,m} − (ℓ+1) α⁻_ℓ Y_{ℓ-1,m}
    # with α⁺_ℓ = √[((ℓ+1)²−m²)/((2ℓ+1)(2ℓ+3))], α⁻_ℓ = √[(ℓ²−m²)/((2ℓ−1)(2ℓ+1))]
    # (these α are the orthonormal recurrence coefficients, == sin_theta_coupling).

    # A⁺: coupling to ℓ+1  =  +ℓ α⁺_ℓ
    A_plus = 0.0
    if ℓ + 1 >= abs(m)
        num = (ℓ + 1 + m) * (ℓ + 1 - m)
        den = (2ℓ + 1) * (2ℓ + 3)
        if num >= 0 && den > 0
            A_plus = ℓ * sqrt(num / den)
        end
    end

    # A⁻: coupling to ℓ-1  =  −(ℓ+1) α⁻_ℓ
    A_minus = 0.0
    if ℓ - 1 >= abs(m) && ℓ > 0
        num = (ℓ + m) * (ℓ - m)
        den = (2ℓ - 1) * (2ℓ + 1)
        if num >= 0 && den > 0
            A_minus = -(ℓ + 1) * sqrt(num / den)
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
    am = abs(m)
    (L < am || ℓ < am) && return 0.0
    # Opposite parity ⇒ integrand odd under x→−x ⇒ exactly zero.
    (L + ℓ) % 2 != 0 && return 0.0

    # Exact ⟨Y_Lm | 1/sinθ | Y_ℓm⟩ = ∫₀^π P̄_Lm(cosθ) P̄_ℓm(cosθ) dθ — the 1/sinθ
    # cancels the sinθ of dΩ. This is the Gauss–Chebyshev G(a,b) used by
    # _dtheta_sphere_projection, so it carries the SAME orthonormal SH convention
    # as the cosθ / sinθ∂θ operators. Couples all same-parity L (0, ±2, ±4, …).
    lmax_needed = max(L, ℓ)
    Nq = lmax_needed + 2
    s = 0.0
    for j in 1:Nq
        x = cos(π * (j - 0.5) / Nq)
        P = _orthonormal_plm(lmax_needed, am, x)
        s += P[L + 1] * P[ℓ + 1]
    end
    return (π / Nq) * s
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

    # `m_bs` may be negative: the sin(|m|φ) partner of the temperature mode. All
    # angular operators / ℓ-ranges depend only on |m| (`am`); the signed `m_bs` is
    # used solely for the dictionary keys. By φ-rotation symmetry the sin partner's
    # meridional flow has the same radial profile as the cosine mode, so `T(am)`
    # drives both. For m_bs ≥ 0, am == m_bs and this path is bit-identical.
    am = abs(m_bs)

    # Non-dimensional parameters. Match the validated φ-thermal-wind convention
    # (solve_thermal_wind_coupled!): the curled geostrophic balance is divided by
    # 2Ω so the (ẑ·∇) operator is bare and the buoyancy carries the 1/(2Ω) (Ω=1 → /2).
    two_omega = one(T)
    buoyancy_factor = Ra * E^2 / (2 * Pr)

    # Boundary indices
    idx_inner = abs(r[1] - r_i) < abs(r[Nr] - r_i) ? 1 : Nr
    idx_outer = idx_inner == 1 ? Nr : 1

    # Number of ℓ modes: |m_bs|, |m_bs|+1, ..., lmax_bs
    n_ell = lmax_bs - am + 1
    if n_ell <= 0
        return nothing
    end

    # =========================================================================
    # θ-thermal-wind balance solved as a coupled Galerkin system, identical in
    # structure to the validated solve_thermal_wind_coupled!:
    #     Σ_L [ A_KL dU_L/dr − (1/r) B_KL U_L ] = F_K ,
    # with A_KL = ⟨Y_Km|cosθ|Y_Lm⟩, B_KL = ⟨Y_Km|sinθ∂θ|Y_Lm⟩ the orthonormal
    # (ẑ·∇) projection (cos_theta_coupling / theta_derivative_coupling), and the
    # forcing F_K the projection of the (1/sinθ)∂_φ buoyancy. The operator is
    # FIRST-ORDER in radius, so exactly ONE radial BC per mode (inner); no
    # diagonal regularization (the old block-build applied two BCs + a Tikhonov
    # diagonal and did not satisfy the PDE — see test/audit_fixes.jl residual test).
    # =========================================================================
    r2 = r .* r
    modes = am:lmax_bs

    # cosθ and sinθ∂θ projection matrices: column L → rows L±1.
    A_mat = zeros(T, n_ell, n_ell)   # ⟨Y_Km|cosθ|Y_Lm⟩
    B_mat = zeros(T, n_ell, n_ell)   # ⟨Y_Km|sinθ∂θ|Y_Lm⟩
    for (k2, L) in enumerate(modes)
        b_minus, b_plus = cos_theta_coupling(L, am)
        B_minus, B_plus, _ = theta_derivative_coupling(L, am)
        kp = (L + 1) - am + 1            # row K = L+1
        if kp <= n_ell
            A_mat[kp, k2] = T(b_plus);  B_mat[kp, k2] = T(B_plus)
        end
        km = (L - 1) - am + 1            # row K = L-1
        if km >= 1
            A_mat[km, k2] = T(b_minus); B_mat[km, k2] = T(B_minus)
        end
    end

    # Forcing F_K(r) = -(Ra E²/(2 Pr r_o)) · |m| · Σ_ℓ ⟨Y_Km|1/sinθ|Y_ℓm⟩ Θ̄_ℓ,
    # built directly into the flat RHS vector (no F matrix / permutedims copy).
    # (Linear gravity g=r/r_o cancels the explicit 1/r → constant 1/r_o.)
    n_total = n_ell * Nr
    F_vec = zeros(T, n_total)
    for (k, K) in enumerate(modes)
        base = (k - 1) * Nr
        for ℓ in modes
            haskey(theta_coeffs, (ℓ, m_bs)) || continue
            g = inv_sin_theta_gaunt(K, ℓ, am)
            abs(g) > eps(T) || continue
            c = -buoyancy_factor * T(am) * g / r_o
            θℓ = theta_coeffs[(ℓ, m_bs)]
            @inbounds for i in 1:Nr
                F_vec[base + i] += c * θℓ[i]
            end
        end
    end

    # Assemble L_op = A ⊗ D1 − diag(1/r)·B ⊗ I.
    L_op = zeros(T, n_total, n_total)
    for k1 in 1:n_ell, k2 in 1:n_ell
        rs = (k1 - 1) * Nr;  cs = (k2 - 1) * Nr
        if abs(A_mat[k1, k2]) > eps(T)
            a = A_mat[k1, k2]
            @inbounds for j in 1:Nr, i in 1:Nr
                L_op[rs + i, cs + j] += a * D1[i, j]
            end
        end
        if abs(B_mat[k1, k2]) > eps(T)
            b = B_mat[k1, k2]
            @inbounds for i in 1:Nr
                L_op[rs + i, cs + i] -= b / r[i]
            end
        end
    end

    # One radial BC per mode at the inner boundary (first-order operator).
    for k in 1:n_ell
        bc = (k - 1) * Nr + idx_inner
        L_op[bc, :] .= zero(T)
        if mechanical_bc == :no_slip
            L_op[bc, bc] = one(T)
        elseif mechanical_bc == :stress_free
            cs = (k - 1) * Nr
            L_op[bc, (cs + 1):(cs + Nr)] .= D1[idx_inner, :]
            L_op[bc, bc] -= one(T) / r[idx_inner]
        else
            throw(ArgumentError("mechanical_bc must be :no_slip or :stress_free, got :$mechanical_bc"))
        end
        F_vec[bc] = zero(T)
    end

    F_lu = lu(L_op; check=false)
    rcond = issuccess(F_lu) ?
        LinearAlgebra.LAPACK.gecon!('1', F_lu.factors, opnorm(L_op, 1)) : zero(real(T))
    if rcond < eps(real(T))
        # Singular (e.g. a geostrophic null mode): minimum-norm least-squares.
        u_theta_vec = pinv(L_op) * F_vec
        @warn "Coupled meridional u_θ system is singular; returning the \
               minimum-norm solution." rcond m_bs lmax_bs maxlog=1
    else
        u_theta_vec = F_lu \ F_vec
    end

    # Extract u_θ for each mode
    for (k, ℓ) in enumerate(modes)
        utheta_ℓ = u_theta_vec[((k - 1) * Nr + 1):(k * Nr)]
        utheta_coeffs[(ℓ, m_bs)] = utheta_ℓ
        dutheta_dr_coeffs[(ℓ, m_bs)] = D1 * utheta_ℓ
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

    A_ur = copy(D1)
    A_ur[idx_inner, :] .= zero(T)
    A_ur[idx_inner, idx_inner] = one(T)
    A_ur[idx_outer, :] .= zero(T)
    A_ur[idx_outer, idx_outer] = one(T)
    A_ur_lu = lu(A_ur)

    source_ur = zeros(T, Nr)
    rhs_ur = similar(r)
    r2_ur = similar(r)

    for (i_L, L) in enumerate(am:lmax_bs)
        # Source term for u_r at mode L
        fill!(source_ur, zero(T))

        # Contribution from ∂(sinθ u_θ)/∂θ projected onto Y_Lm
        # sinθ u_θ Y_ℓm has ∂/∂θ that couples via d±
        for (i_ell, ℓ) in enumerate(am:lmax_bs)
            if !haskey(utheta_coeffs, (ℓ, m_bs))
                continue
            end
            utheta_ell = utheta_coeffs[(ℓ, m_bs)]

            # Coupling coefficient for ∂(sinθ Y_ℓm)/∂θ at mode L
            # Using: ∂(sinθ Y)/∂θ = cosθ Y + sinθ ∂Y/∂θ
            d_Lell = zero(T)
            if L == ℓ + 1
                _, C_plus = cos_theta_coupling(ℓ, am)
                _, A_plus, _ = theta_derivative_coupling(ℓ, am)
                d_Lell = T(C_plus) + T(A_plus)
            elseif L == ℓ - 1
                C_minus, _ = cos_theta_coupling(ℓ, am)
                A_minus, _, _ = theta_derivative_coupling(ℓ, am)
                d_Lell = T(C_minus) + T(A_minus)
            elseif L == ℓ
                # Diagonal contribution from cosθ (zero) and sinθ∂/∂θ (small)
                _, _, A_diag = theta_derivative_coupling(ℓ, am)
                d_Lell = T(A_diag)
            end

            if abs(d_Lell) > eps(T)
                source_ur .-= d_Lell .* utheta_ell ./ r
            end
        end

        # Contribution from u_φ term (if present)
        if haskey(uphi_coeffs, (L, m_bs))
            uphi_L = uphi_coeffs[(L, m_bs)]
            inv_sin_LL = inv_sin_theta_gaunt(L, L, am)
            source_ur .-= T(am) .* inv_sin_LL .* uphi_L ./ r
        end

        # Solve: ∂(r² u_r)/∂r = r² × source_ur
        # → D1 × (r² u_r) = r² × source_ur
        # With BC: u_r = 0 at boundaries

        @. rhs_ur = r2 * source_ur
        rhs_ur[idx_inner] = zero(T)
        rhs_ur[idx_outer] = zero(T)

        # Solve for r² u_r
        ldiv!(r2_ur, A_ur_lu, rhs_ur)

        # Extract u_r
        ur_L = r2_ur ./ r2
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
    # Same convention as solve_meridional_coupled! / solve_thermal_wind_coupled!:
    # 2Ω divided out (bare operator), buoyancy carries 1/(2Ω) (Ω=1 → /2).
    two_omega = one(T)
    buoyancy_factor = Ra * E^2 / (2 * Pr)

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

    # m ≠ 0: Solve simplified θ-thermal wind equation. Signed m_bs: the sin(|m|φ)
    # partner (m_bs<0) shares the cosine mode's radial profile by φ-rotation symmetry;
    # angular factors use |m| (`am`), the signed key stores cos (+m) vs sin (−m).
    for m_bs in (-mmax_bs:mmax_bs)
        m_bs == 0 && continue
        am = abs(m_bs)
        for ℓ in am:lmax_bs
            # Check for temperature forcing
            if !haskey(theta_coeffs, (ℓ, m_bs))
                ur_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                utheta_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                dur_dr_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                dutheta_dr_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                continue
            end

            T_lm = theta_coeffs[(ℓ, m_bs)]
            if _maxabs(T_lm) < eps(T) * 100
                ur_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                utheta_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                dur_dr_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                dutheta_dr_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                continue
            end

            # Effective factors for diagonal approximation
            # sinθ_eff: effective value of 1/sinθ for mode (ℓ,m)
            inv_sin_eff = one(T) + T(am^2) / T(max(ℓ * (ℓ + 1), 1))

            # β_eff: effective z-derivative factor (latitude average of cosθ)
            # For geostrophic flow, use the characteristic value
            β_eff = sqrt(T(ℓ) / T(ℓ + 1))

            # RHS forcing: -(Ra E²/(2Pr r_o)) × im × T̄_ℓm × inv_sin_eff
            # (linear gravity g=r/r_o cancels the explicit 1/r → 1/r_o)
            forcing = -buoyancy_factor .* T(am) .* T_lm .* inv_sin_eff ./ r_o

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
            elseif mechanical_bc == :stress_free
                A_mat[idx_inner, :] .= zero(T)
                A_mat[idx_inner, :] .= D1[idx_inner, :]
                A_mat[idx_inner, idx_inner] -= one(T) / r[idx_inner]
                forcing[idx_inner] = zero(T)

                A_mat[idx_outer, :] .= zero(T)
                A_mat[idx_outer, :] .= D1[idx_outer, :]
                A_mat[idx_outer, idx_outer] -= one(T) / r[idx_outer]
                forcing[idx_outer] = zero(T)
            else
                throw(ArgumentError("mechanical_bc must be :no_slip or :stress_free, got :$mechanical_bc"))
            end

            # Solve for u_θ
            utheta_lm = A_mat \ forcing

            # Compute u_r from continuity: ∂(r² u_r)/∂r + [angular terms] = 0
            # Simplified: u_r ~ -(r/ℓ(ℓ+1)) × (angular derivative of u_θ)
            # Using c_θ factor as proxy for angular derivative magnitude
            ell_factor = T(ℓ * (ℓ + 1))
            c_theta = ell_factor > 0 ? sqrt(max(ell_factor - T(am^2), zero(T))) / sqrt(ell_factor) : zero(T)

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
        for m_bs in (-mmax_bs:mmax_bs)
            for ℓ in abs(m_bs):lmax_bs
                ur_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                utheta_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                dur_dr_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
                dutheta_dr_coeffs[(ℓ, m_bs)] = zeros(T, Nr)
            end
        end
        return nothing
    end

    if use_full_coupling
        # Full coupled solver: solves the complete block-tridiagonal system for each
        # azimuthal wavenumber m separately. Signed range: m_bs<0 develops the sin(|m|φ)
        # meridional partner (m_bs=0 returns zeros; ±|m| share a radial profile).
        for m_bs in (-mmax_bs:mmax_bs)
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
                                     ur_coeffs, dur_dr_coeffs,
                                     utheta_coeffs, uphi_coeffs,
                                     lmax_bs, mmax_bs, r)

Full advection ū·∇T̄ in spectral space, computed correctly via the divergence
form `∇·(ūT̄)` (valid for incompressible ū) using a vector-spherical-harmonic
transform — see `vecsh_advection`. This is aliasing-free and captures the full
triadic (Gaunt) coupling, replacing the former approximate term-split (Term 1
diagonal-only, Term 2 ℓ±1-only, Term 3 spurious "empirical" factors).

The basic state is stored in the full real-SH `±m` basis (`cos mφ` at `+m`,
`sin mφ` at `-m`); both signs are computed and returned, so the forcing is the
exact projection of `∇·(ūT̄)` including the sin(mφ) contributions that the
self-consistent iteration retains.
"""
function compute_full_advection_spectral(
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dtheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    ur_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dur_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    utheta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    lmax_bs::Int, mmax_bs::Int,
    r::Vector{T}
) where T<:Real
    # The basic state is stored in the no-factorial normalization; vecsh_advection
    # works in the orthonormal (full N_ℓm) basis. Convert inputs in, output back
    # (m=0 conversion is identity, so the validated axisymmetric path is unchanged).
    F_orth = vecsh_advection(
        _sh_rescale(theta_coeffs, +1), _sh_rescale(dtheta_dr_coeffs, +1),
        _sh_rescale(ur_coeffs, +1), _sh_rescale(dur_dr_coeffs, +1),
        _sh_rescale(utheta_coeffs, +1), _sh_rescale(uphi_coeffs, +1),
        lmax_bs, mmax_bs, r)
    return _sh_rescale(F_orth, -1)
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
        verbose::Bool = false,
        coupled_thermal_wind::Bool = true
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
- `coupled_thermal_wind` : Use full coupled thermal-wind solve (default: true)

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
    verbose::Bool = false,
    coupled_thermal_wind::Bool = true
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
        outer_fluxes=outer_fluxes_T,
        coupled_thermal_wind=coupled_thermal_wind
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

    # Initialize to zero for all modes. Negative m holds the sin(|m|φ) partner so
    # the self-consistent state can develop the φ-advection (zonal-drift) physics.
    for ℓ in 0:lmax_bs
        for m in -min(ℓ, mmax_bs):min(ℓ, mmax_bs)
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

        # Store previous temperature for convergence check. The coefficient dict
        # holds plain numeric vectors, so copying each vector is equivalent to
        # `deepcopy` but skips its identity-tracking overhead.
        theta_prev = Dict(k => copy(v) for (k, v) in theta_coeffs)

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
            ur_coeffs, dur_dr_coeffs, utheta_coeffs, uphi_coeffs,
            lmax_bs, mmax_bs, r
        )

        # Scale by 1/κ
        for (key, val) in advection_source
            val ./= κ_eff
        end

        # ---------------------------------------------------------------------
        # Step 4: Solve Poisson equation ∇²T̄ = S with boundary conditions
        # ---------------------------------------------------------------------
        zero_forcing = zeros(T, Nr)
        for ℓ in 0:lmax_bs
            for m in -min(ℓ, mmax_bs):min(ℓ, mmax_bs)  # ±m: sin partners develop via φ-advection
                # Get forcing for this mode (zero if no advection). The Poisson
                # operator is m-independent (eigenvalue ℓ(ℓ+1)), so m<0 solves
                # identically with the sin-mode forcing.
                forcing = get(advection_source, (ℓ, m), zero_forcing)

                # Get boundary conditions
                inner_val, outer_val, bc_type = get(bc_values, (ℓ, m), (zero(T), zero(T), :fixed_temperature))

                # Only solve if there's forcing OR this mode has BC amplitude
                has_bc = abs(outer_val) > eps(T) * 100 || (ℓ == 0 && m == 0)
                has_forcing = _maxabs(forcing) > eps(T) * 100

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
        # ±m: cos (m>0) and sin (m<0) partners. The thermal-wind solvers expect a
        # non-negative wavenumber, so call them with |m| (the coupling depends only
        # on |m|) and store the result under the signed m.
        for m_bs in -mmax_bs:mmax_bs
            am = abs(m_bs)
            # Extract temperature modes for this (signed) m
            theta_m = Dict{Int, Vector{T}}()
            for ℓ in am:lmax_bs
                if haskey(theta_coeffs, (ℓ, m_bs))
                    theta_m[ℓ] = theta_coeffs[(ℓ, m_bs)]
                end
            end

            if isempty(theta_m) || all(_maxabs(v) < eps(T) * 100 for v in values(theta_m))
                continue
            end

            # Initialize velocity storage for this m
            uphi_m = Dict{Int, Vector{T}}(ℓ => zeros(T, Nr) for ℓ in 0:lmax_bs)
            duphi_dr_m = Dict{Int, Vector{T}}(ℓ => zeros(T, Nr) for ℓ in 0:lmax_bs)

            # Solve thermal wind (use |m| — coupling depends only on |m|)
            if coupled_thermal_wind
                # Full coupled solver (no diagonal approximation)
                solve_thermal_wind_coupled!(uphi_m, duphi_dr_m, theta_m, am,
                                            cd, r_i, r_o, Ra, Pr;
                                            mechanical_bc=mechanical_bc,
                                            E=E, lmax=lmax_bs + 1)
            else
                # Diagonal approximation
                solve_thermal_wind_balance_3d!(uphi_m, duphi_dr_m, theta_m, am,
                                               cd, r_i, r_o, Ra, Pr;
                                               mechanical_bc=mechanical_bc,
                                               E=E)
            end

            # Copy results to storage under the signed m
            for ℓ in 0:lmax_bs
                if haskey(uphi_m, ℓ) && _maxabs(uphi_m[ℓ]) > eps(T) * 100
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
                change = _maxabsdiff(theta_new, theta_old)
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
        for m in -min(ℓ, mmax_bs):min(ℓ, mmax_bs)
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
- `coupled_thermal_wind` : Use full coupled thermal-wind solve (default: true)

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
                                    verbose::Bool=false,
                                    coupled_thermal_wind::Bool=true)

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
                          lmax_bs=_lmax,
                          coupled_thermal_wind=coupled_thermal_wind), nothing
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
            verbose=verbose,
            coupled_thermal_wind=coupled_thermal_wind
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
            verbose=verbose,
            coupled_thermal_wind=coupled_thermal_wind
        )
    end
end
