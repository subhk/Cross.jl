# =============================================================================
#  Basic State for Onset of Convection
#
#  Implements both axisymmetric and non-axisymmetric basic states with:
#  - Temperature variations: θ̄(r,θ,φ)
#  - Thermal wind-balanced flows: ū(r,θ,φ)
#
#  Two implementations:
#  1. BasicState: Axisymmetric (m=0 only), for standard onset
#  2. BasicState3D: Non-axisymmetric (multiple m), for tri-global analysis
# =============================================================================

using Parameters
using LinearAlgebra

"""
    BasicState{T<:Real}

Holds the axisymmetric (m=0) basic state for linear stability analysis.

The basic state consists of:
- Temperature: θ̄(r,θ) = Σ_ℓ θ̄_ℓ0(r) Y_ℓ0(θ)
- Zonal flow: ū_φ(r,θ) = Σ_ℓ ū_φ,ℓ0(r) Y_ℓ0(θ)
- No meridional flow: ū_r = ū_θ = 0

Fields:
- `lmax_bs::Int` - Maximum spherical harmonic degree for basic state
- `Nr::Int` - Number of radial collocation points
- `r::Vector{T}` - Radial collocation points
- `theta_coeffs::Dict{Int,Vector{T}}` - Temperature coefficients θ̄_ℓ0(r) for each ℓ
- `uphi_coeffs::Dict{Int,Vector{T}}` - Zonal flow coefficients ū_φ,ℓ0(r) for each ℓ
- `dtheta_dr_coeffs::Dict{Int,Vector{T}}` - Radial derivative ∂θ̄_ℓ0/∂r
- `duphi_dr_coeffs::Dict{Int,Vector{T}}` - Radial derivative ∂ū_φ,ℓ0/∂r
"""
@with_kw struct BasicState{T<:Real}
    lmax_bs::Int
    Nr::Int
    r::Vector{T}
    theta_coeffs::Dict{Int,Vector{T}}
    uphi_coeffs::Dict{Int,Vector{T}}
    dtheta_dr_coeffs::Dict{Int,Vector{T}}
    duphi_dr_coeffs::Dict{Int,Vector{T}}
end


"""
    BasicState3D{T<:Real}

Holds a non-axisymmetric (3D) basic state for tri-global instability analysis.

The basic state has both meridional AND longitudinal variations:
- Temperature: θ̄(r,θ,φ) = Σ_ℓ Σ_m_bs θ̄_ℓm_bs(r) Y_ℓm_bs(θ,φ)
- Velocity components: ū_r, ū_θ, ū_φ from geostrophic balance

This enables studying onset of convection on top of 3D thermal and flow structures,
such as:
- Longitudinally-varying boundary heating
- Zonal jets with wavenumber structure
- Realistic 3D planetary/stellar base states

Fields:
- `lmax_bs::Int` - Maximum spherical harmonic degree
- `mmax_bs::Int` - Maximum azimuthal wavenumber (typically small, e.g., 0-4)
- `Nr::Int` - Number of radial collocation points
- `r::Vector{T}` - Radial collocation points
- `theta_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - θ̄_ℓm(r) indexed by (ℓ,m)
- `ur_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ū_r,ℓm(r)
- `utheta_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ū_θ,ℓm(r)
- `uphi_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ū_φ,ℓm(r)
- `dtheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ∂θ̄_ℓm/∂r
- `dur_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ∂ū_r,ℓm/∂r
- `dutheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ∂ū_θ,ℓm/∂r
- `duphi_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ∂ū_φ,ℓm/∂r

Note: Perturbations on this basic state couple multiple azimuthal modes m simultaneously.
The eigenvalue problem becomes block-coupled across different m values.
"""
@with_kw struct BasicState3D{T<:Real}
    lmax_bs::Int
    mmax_bs::Int
    Nr::Int
    r::Vector{T}
    # Temperature
    theta_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    dtheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    # Velocity components
    ur_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    utheta_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    uphi_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    # Velocity derivatives
    dur_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    dutheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    duphi_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
end


"""
    conduction_basic_state(cd::ChebyshevDiffn{T}, χ::T, lmax_bs::Int) where T

Create a basic state corresponding to pure conduction (no meridional variation).

This is the default basic state with:
- θ̄(r) = conduction profile (only ℓ=0 component)
- ū_φ = 0 (no flow)

Arguments:
- `cd` - Chebyshev differentiation structure
- `χ` - Radius ratio r_i/r_o
- `lmax_bs` - Maximum ℓ for basic state (typically small, e.g., 4)
"""
function conduction_basic_state(cd::ChebyshevDiffn{T}, χ::T, lmax_bs::Int) where T
    r = cd.x
    Nr = length(r)

    # Conduction temperature profile (only ℓ=0 component)
    # θ̄_cond(r) such that ∇²θ̄ = 0 with θ̄(r_i)=1, θ̄(r_o)=0
    # Solution: θ̄(r) = (r_o/r - 1)/(r_o/r_i - 1)
    r_i = χ
    r_o = 1.0
    theta_cond = @. (r_o/r - 1.0)/(r_o/r_i - 1.0)

    # Radial derivative
    dtheta_dr_cond = cd.D1 * theta_cond

    # Initialize dictionaries
    theta_coeffs = Dict{Int,Vector{T}}()
    uphi_coeffs = Dict{Int,Vector{T}}()
    dtheta_dr_coeffs = Dict{Int,Vector{T}}()
    duphi_dr_coeffs = Dict{Int,Vector{T}}()

    # Only ℓ=0 component is non-zero
    # Need to normalize by spherical harmonic coefficient
    # Y_00 = 1/√(4π), so θ̄_00(r) = √(4π) × θ_cond(r)
    theta_coeffs[0] = sqrt(4π) .* theta_cond
    dtheta_dr_coeffs[0] = sqrt(4π) .* dtheta_dr_cond
    uphi_coeffs[0] = zeros(T, Nr)
    duphi_dr_coeffs[0] = zeros(T, Nr)

    # Higher ℓ modes are zero
    for ℓ in 1:lmax_bs
        theta_coeffs[ℓ] = zeros(T, Nr)
        dtheta_dr_coeffs[ℓ] = zeros(T, Nr)
        uphi_coeffs[ℓ] = zeros(T, Nr)
        duphi_dr_coeffs[ℓ] = zeros(T, Nr)
    end

    return BasicState(
        lmax_bs = lmax_bs,
        Nr = Nr,
        r = r,
        theta_coeffs = theta_coeffs,
        uphi_coeffs = uphi_coeffs,
        dtheta_dr_coeffs = dtheta_dr_coeffs,
        duphi_dr_coeffs = duphi_dr_coeffs
    )
end


"""
    meridional_basic_state(cd::ChebyshevDiffn{T}, χ::T, Ra::T, Pr::T,
                          lmax_bs::Int, amplitude::T) where T

Create a basic state with meridional temperature variation at the outer boundary.

The outer boundary temperature varies meridionally:
    θ̄(r_o, θ) = 1 + amplitude × Y_20(θ)

The inner boundary is held at uniform temperature:
    θ̄(r_i, θ) = 0

This represents differential heating (e.g., equator hotter than poles).

The basic state temperature θ̄(r,θ) is found by solving the conduction equation:
    ∇²θ̄ = 0

with these boundary conditions. The zonal flow ū_φ(r,θ) is then computed from
thermal wind balance:
    ∂ū_φ/∂r + ū_φ/r = -(Ra)/(2Pr) × (r/r_o)/(r sin(θ)) × ∂θ̄/∂θ

Arguments:
- `cd` - Chebyshev differentiation structure
- `χ` - Radius ratio r_i/r_o
- `Ra` - Rayleigh number (needed for thermal wind balance)
- `Pr` - Prandtl number
- `lmax_bs` - Maximum ℓ for basic state expansion
- `amplitude` - Amplitude of meridional variation at outer boundary (typically 0.01-0.1)
"""
function meridional_basic_state(cd::ChebyshevDiffn{T}, χ::T, Ra::T, Pr::T,
                               lmax_bs::Int, amplitude::T) where T

    r = cd.x
    Nr = length(r)
    r_i = χ
    r_o = 1.0

    # Initialize dictionaries
    theta_coeffs = Dict{Int,Vector{T}}()
    dtheta_dr_coeffs = Dict{Int,Vector{T}}()
    uphi_coeffs = Dict{Int,Vector{T}}()
    duphi_dr_coeffs = Dict{Int,Vector{T}}()

    # Solve conduction equation ∇²θ̄ = 0 for each ℓ mode
    # For ℓ=0: standard conduction profile
    # For ℓ>0: solve with meridional boundary conditions

    # =========================================================================
    # ℓ=0 mode: Standard conduction (radially symmetric)
    # =========================================================================
    # BC: θ̄_00(r_i) = 0, θ̄_00(r_o) = √(4π) (normalized by Y_00)
    # Solution: θ̄_00(r) = √(4π) × [(r_o/r - 1)/(r_o/r_i - 1)]

    theta_cond = @. (r_o/r - 1.0)/(r_o/r_i - 1.0)
    dtheta_dr_cond = cd.D1 * theta_cond

    theta_coeffs[0] = sqrt(4π) .* theta_cond
    dtheta_dr_coeffs[0] = sqrt(4π) .* dtheta_dr_cond
    uphi_coeffs[0] = zeros(T, Nr)
    duphi_dr_coeffs[0] = zeros(T, Nr)

    # =========================================================================
    # ℓ=2 mode: Meridional variation driven by boundary conditions
    # =========================================================================
    # The scalar Laplacian for ℓ=2 in spherical coordinates is:
    # Sℓ[θ̄_ℓ0] = (1/r²) d/dr(r² dθ̄_ℓ0/dr) - ℓ(ℓ+1)/r² θ̄_ℓ0 = 0
    #
    # This gives: d²θ̄_ℓ0/dr² + (2/r) dθ̄_ℓ0/dr - ℓ(ℓ+1)/r² θ̄_ℓ0 = 0
    #
    # For ℓ=2: d²θ̄/dr² + (2/r) dθ̄/dr - 6/r² θ̄ = 0
    #
    # General solution: θ̄(r) = A r² + B r⁻³
    #
    # BC: θ̄(r_i) = 0, θ̄(r_o) = amplitude / norm_Y20

    norm_Y20 = sqrt(5/(4π))
    ℓ = 2

    # Solve the boundary value problem using the spectral method
    # Lℓ[θ̄] = 0 with BCs

    Lℓ_op = cd.D2 + Diagonal(2 ./ r) * cd.D1 - Diagonal(ℓ*(ℓ+1) ./ r.^2)

    # Build the system with boundary conditions
    # Replace first and last rows with boundary conditions
    A_system = copy(Lℓ_op)
    A_system[1, :] .= 0.0
    A_system[1, 1] = 1.0  # θ̄(r_i) = 0
    A_system[end, :] .= 0.0
    A_system[end, end] = 1.0  # θ̄(r_o) = amplitude / norm_Y20

    # Right-hand side
    rhs = zeros(T, Nr)
    rhs[1] = 0.0  # Inner boundary
    rhs[end] = amplitude / norm_Y20  # Outer boundary (meridional variation!)

    # Solve for θ̄_20(r)
    theta_20 = A_system \ rhs
    dtheta_20_dr = cd.D1 * theta_20

    theta_coeffs[2] = theta_20
    dtheta_dr_coeffs[2] = dtheta_20_dr
    uphi_coeffs[2] = zeros(T, Nr)
    duphi_dr_coeffs[2] = zeros(T, Nr)

    # =========================================================================
    # Higher ℓ modes: zero (no higher-order boundary variations)
    # =========================================================================
    for ℓ in [1, 3, 4]
        if ℓ <= lmax_bs && ℓ != 2
            theta_coeffs[ℓ] = zeros(T, Nr)
            dtheta_dr_coeffs[ℓ] = zeros(T, Nr)
            uphi_coeffs[ℓ] = zeros(T, Nr)
            duphi_dr_coeffs[ℓ] = zeros(T, Nr)
        end
    end

    # =========================================================================
    # Solve thermal wind balance for ū_φ
    # =========================================================================
    solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                cd, r_i, r_o, Ra, Pr)

    return BasicState(
        lmax_bs = lmax_bs,
        Nr = Nr,
        r = r,
        theta_coeffs = theta_coeffs,
        uphi_coeffs = uphi_coeffs,
        dtheta_dr_coeffs = dtheta_dr_coeffs,
        duphi_dr_coeffs = duphi_dr_coeffs
    )
end


# Helper: coefficients for expanding derivatives of Legendre polynomials.
#
# Returns a vector `deriv_maps` where `deriv_maps[ℓ]` is a dictionary mapping
# target degree L to the coefficient c_{ℓ,L} in
#     P_ℓ'(x) = Σ c_{ℓ,L} P_L(x)
# with L ranging over ℓ-1, ℓ-3, … (same parity as ℓ-1).
function legendre_derivative_coefficients(lmax::Int)
    maps = Dict{Int, Dict{Int,Float64}}()
    maps[0] = Dict{Int,Float64}()           # P₀' = 0
    if lmax >= 1
        maps[1] = Dict(0 => 1.0)            # P₁' = P₀
    end

    for ℓ in 2:lmax
        coeffs = Dict{Int,Float64}()
        coeffs[ℓ-1] = (2ℓ - 1) * 1.0        # (2ℓ-1) P_{ℓ-1}

        for (k, v) in maps[ℓ-2]
            coeffs[k] = get(coeffs, k, 0.0) + v
        end

        maps[ℓ] = coeffs
    end

    return maps
end


"""
    solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                cd, r_i, r_o, Ra, Pr)

Solve the thermal wind balance equation to compute zonal flow coefficients.

The thermal wind equation in non-dimensional form is:
    ∂ū_φ/∂r + ū_φ/r = -(Ra)/(2Pr) × (r/r_o)/(r sin(θ)) × ∂θ̄/∂θ

This is solved as a system of ODEs for the spectral coefficients ū_φ,ℓ0(r).

Modifies `uphi_coeffs` and `duphi_dr_coeffs` in place.
"""
function solve_thermal_wind_balance!(uphi_coeffs::Dict{Int,Vector{T}},
                                     duphi_dr_coeffs::Dict{Int,Vector{T}},
                                     theta_coeffs::Dict{Int,Vector{T}},
                                     cd::ChebyshevDiffn{T},
                                     r_i::T, r_o::T, Ra::T, Pr::T) where T

    r = cd.x
    Nr = length(r)

    lmax = maximum(keys(theta_coeffs))
    deriv_maps = legendre_derivative_coefficients(lmax)

    # Compute coefficients of (1/sinθ) ∂θ̄/∂θ expanded in Y_ℓ0
    grad_coeffs = Dict{Int, Vector{T}}()
    four_T = T(4)
    pi_T = T(pi)

    for (ℓ, coeff_vec) in theta_coeffs
        if ℓ == 0
            continue
        end
        deriv_map = get(deriv_maps, ℓ, Dict{Int,Float64}())
        if isempty(deriv_map)
            continue
        end

        norm_source = sqrt(T(2ℓ + 1) / (four_T * pi_T))

        for (L, c_val) in deriv_map
            if !haskey(grad_coeffs, L)
                grad_coeffs[L] = zeros(T, Nr)
            end
            norm_target = sqrt(T(2L + 1) / (four_T * pi_T))
            factor = -norm_source / norm_target * T(c_val)
            grad_coeffs[L] .+= factor .* coeff_vec
        end
    end

    prefactor = -(Ra / (2 * Pr)) / r_o

    for L in keys(grad_coeffs)
        forcing = prefactor .* (r.^2) .* grad_coeffs[L]
        r_uphi = zeros(T, Nr)
        for i in 2:Nr
            dr = r[i] - r[i-1]
            r_uphi[i] = r_uphi[i-1] + 0.5 * (forcing[i-1] + forcing[i]) * dr
        end
        uphi_L = r_uphi ./ r
        uphi_L[1] = zero(T)
        uphi_L[end] = zero(T)

        uphi_coeffs[L] = uphi_L
        duphi = cd.D1 * uphi_L
        duphi[1] = zero(T)
        duphi[end] = zero(T)
        duphi_dr_coeffs[L] = duphi
    end

    # Zero out modes without forcing
    grad_modes = collect(keys(grad_coeffs))
    for ℓ in collect(keys(uphi_coeffs))
        if !(ℓ in grad_modes)
            uphi_coeffs[ℓ] .= zero(T)
            duphi_dr_coeffs[ℓ] .= zero(T)
        end
    end

    return nothing
end


"""
    evaluate_basic_state(bs::BasicState{T}, r_eval::T, theta_eval::T) where T

Evaluate the basic state at a given (r, θ) point.

Returns:
- `theta_bar` - Temperature θ̄(r,θ)
- `uphi_bar` - Zonal velocity ū_φ(r,θ)
- `dtheta_dr` - Radial derivative ∂θ̄/∂r
- `dtheta_dtheta` - Meridional derivative ∂θ̄/∂θ
- `duphi_dr` - Radial derivative ∂ū_φ/∂r
- `duphi_dtheta` - Meridional derivative ∂ū_φ/∂θ
"""
function evaluate_basic_state(bs::BasicState{T}, r_eval::T, theta_eval::T) where T
    # This would require interpolation and spherical harmonic evaluation
    # For now, return zeros as placeholder
    # TODO: Implement proper evaluation using spherical harmonics

    return (
        theta_bar = zero(T),
        uphi_bar = zero(T),
        dtheta_dr = zero(T),
        dtheta_dtheta = zero(T),
        duphi_dr = zero(T),
        duphi_dtheta = zero(T)
    )
end


# =============================================================================
#  Non-Axisymmetric (3D) Basic States
# =============================================================================

"""
    nonaxisymmetric_basic_state(cd::ChebyshevDiffn{T}, χ::T, Ra::T, Pr::T,
                                lmax_bs::Int, mmax_bs::Int,
                                amplitudes::Dict{Tuple{Int,Int},T}) where T

Create a 3D basic state with both meridional and longitudinal temperature variations.

The boundary temperature varies in both latitude and longitude:
    θ̄(r_o, θ, φ) = 1 + Σ_{ℓ,m} amplitude_{ℓm} × Y_ℓm(θ,φ)
    θ̄(r_i, θ, φ) = 0

This represents fully 3D differential heating scenarios, such as:
- Longitudinally-varying solar heating
- Zonal wavenumber patterns in thermal forcing
- Realistic planetary/stellar boundary conditions

The interior temperature θ̄(r,θ,φ) satisfies ∇²θ̄ = 0 with these BCs.

The velocity field ū(r,θ,φ) is computed from simplified thermal wind balance
(assuming ū_r = ū_θ = 0, only ū_φ ≠ 0 as an approximation).

Arguments:
- `cd` - Chebyshev differentiation structure
- `χ` - Radius ratio r_i/r_o
- `Ra` - Rayleigh number (for thermal wind balance)
- `Pr` - Prandtl number
- `lmax_bs` - Maximum ℓ for basic state
- `mmax_bs` - Maximum m for basic state (e.g., 0-4)
- `amplitudes` - Dict{(ℓ,m) => amplitude} specifying boundary temperature modes

Example:
    amplitudes = Dict(
        (2,0) => 0.1,   # Meridional Y_20 component
        (2,2) => 0.05   # Longitudinal Y_22 component
    )
    bs3d = nonaxisymmetric_basic_state(cd, χ, Ra, Pr, 4, 2, amplitudes)
"""
function nonaxisymmetric_basic_state(cd::ChebyshevDiffn{T}, χ::T, Ra::T, Pr::T,
                                     lmax_bs::Int, mmax_bs::Int,
                                     amplitudes::Dict{Tuple{Int,Int},T}) where T

    r = cd.x
    Nr = length(r)
    r_i = χ
    r_o = 1.0

    # Initialize all coefficient dictionaries
    theta_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    dtheta_dr_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    ur_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    utheta_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    uphi_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    dur_dr_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    dutheta_dr_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    duphi_dr_coeffs = Dict{Tuple{Int,Int},Vector{T}}()

    # =========================================================================
    # Solve ∇²θ̄ = 0 for each (ℓ,m) mode
    # =========================================================================

    for ℓ in 0:lmax_bs
        for m in 0:min(ℓ, mmax_bs)
            # Get amplitude for this mode (default to 0 if not specified)
            amp = get(amplitudes, (ℓ,m), zero(T))

            if ℓ == 0 && m == 0
                # ℓ=0, m=0: Radial conduction profile
                theta_cond = @. (r_o/r - 1.0)/(r_o/r_i - 1.0)
                dtheta_dr_cond = cd.D1 * theta_cond

                # Normalize by Y_00 = 1/√(4π)
                theta_coeffs[(0,0)] = sqrt(4π) .* theta_cond
                dtheta_dr_coeffs[(0,0)] = sqrt(4π) .* dtheta_dr_cond

            elseif amp != 0
                # Non-zero amplitude: Solve boundary value problem
                # d²θ̄_ℓm/dr² + (2/r) dθ̄_ℓm/dr - ℓ(ℓ+1)/r² θ̄_ℓm = 0
                # BC: θ̄_ℓm(r_i) = 0, θ̄_ℓm(r_o) = amp / norm(Y_ℓm)

                # Construct Laplacian operator for this ℓ
                Lℓ_op = cd.D2 + Diagonal(2 ./ r) * cd.D1 - Diagonal(ℓ*(ℓ+1) ./ r.^2)

                # Apply boundary conditions
                A_system = copy(Lℓ_op)
                A_system[1, :] .= 0.0
                A_system[1, 1] = 1.0  # θ̄(r_i) = 0
                A_system[end, :] .= 0.0
                A_system[end, end] = 1.0  # θ̄(r_o) = amp / norm(Y_ℓm)

                # Normalization of Y_ℓm (Schmidt semi-normalized)
                # For m=0: norm = sqrt((2ℓ+1)/(4π))
                # For m≠0: norm = sqrt((2ℓ+1)/(4π) × 2)
                norm_Ylm = m == 0 ? sqrt((2ℓ+1)/(4π)) : sqrt((2ℓ+1)/(4π) * 2)

                # Right-hand side
                rhs = zeros(T, Nr)
                rhs[1] = 0.0
                rhs[end] = amp / norm_Ylm

                # Solve for θ̄_ℓm(r)
                theta_lm = A_system \ rhs
                dtheta_lm_dr = cd.D1 * theta_lm

                theta_coeffs[(ℓ,m)] = theta_lm
                dtheta_dr_coeffs[(ℓ,m)] = dtheta_lm_dr

            else
                # Zero amplitude: Initialize to zero
                theta_coeffs[(ℓ,m)] = zeros(T, Nr)
                dtheta_dr_coeffs[(ℓ,m)] = zeros(T, Nr)
            end

            # Initialize velocity components to zero (will be filled by thermal wind)
            ur_coeffs[(ℓ,m)] = zeros(T, Nr)
            utheta_coeffs[(ℓ,m)] = zeros(T, Nr)
            uphi_coeffs[(ℓ,m)] = zeros(T, Nr)
            dur_dr_coeffs[(ℓ,m)] = zeros(T, Nr)
            dutheta_dr_coeffs[(ℓ,m)] = zeros(T, Nr)
            duphi_dr_coeffs[(ℓ,m)] = zeros(T, Nr)
        end
    end

    # =========================================================================
    # Solve simplified thermal wind balance for ū_φ
    # =========================================================================
    # NOTE: For non-axisymmetric basic states, the full geostrophic balance
    # would require solving for all velocity components (ū_r, ū_θ, ū_φ).
    # Here we use a simplified approach: assume ū_r = ū_θ = 0 and solve
    # thermal wind for ū_φ only. This is an approximation valid for
    # small amplitude non-axisymmetric perturbations.

    solve_thermal_wind_3d!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                           cd, r_i, r_o, Ra, Pr, lmax_bs, mmax_bs)

    return BasicState3D(
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
end


"""
    solve_thermal_wind_3d!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                           cd, r_i, r_o, Ra, Pr, lmax_bs, mmax_bs)

Solve thermal wind balance for non-axisymmetric basic state.

This is a simplified version that assumes:
- Only ū_φ is non-zero (ū_r = ū_θ = 0)
- Each (ℓ,m) temperature mode drives the same (ℓ,m) velocity mode

The thermal wind equation for each mode is:
    ∂ū_φ,ℓm/∂r + ū_φ,ℓm/r = -C_ℓm × (r/r_o) × θ̄_ℓm(r)

where C_ℓm are coupling coefficients derived from spherical harmonic properties.

Modifies uphi_coeffs and duphi_dr_coeffs in place.
"""
function solve_thermal_wind_3d!(uphi_coeffs::Dict{Tuple{Int,Int},Vector{T}},
                                duphi_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}},
                                theta_coeffs::Dict{Tuple{Int,Int},Vector{T}},
                                cd::ChebyshevDiffn{T},
                                r_i::T, r_o::T, Ra::T, Pr::T,
                                lmax_bs::Int, mmax_bs::Int) where T

    r = cd.x
    Nr = length(r)

    # Loop over all (ℓ,m) modes
    for ℓ in 0:lmax_bs
        for m in 0:min(ℓ, mmax_bs)
            # Skip if temperature coefficient is negligible
            if !haskey(theta_coeffs, (ℓ,m)) || maximum(abs.(theta_coeffs[(ℓ,m)])) < 1e-14
                continue
            end

            # Coupling coefficient for thermal wind (simplified diagonal approximation)
            # For ℓ=2, m=0: C ≈ -2√(5/3) (as before)
            # For general (ℓ,m): Approximate scaling
            if ℓ == 2 && m == 0
                coupling = -2.0 * sqrt(5.0/3.0)
            elseif ℓ >= 2
                # Simplified scaling (diagonal approximation)
                coupling = -Float64(ℓ) * sqrt(Float64(2ℓ+1)/3.0)
            else
                coupling = 0.0
            end

            if abs(coupling) < 1e-14
                continue
            end

            C = (Ra / (2.0 * Pr)) * coupling

            # Thermal wind ODE: ∂(r ū_φ)/∂r = -C × (r²/r_o) × θ̄_ℓm(r)
            rhs = -C .* (r.^2 ./ r_o) .* theta_coeffs[(ℓ,m)]

            # Integrate from r_i with BC: ū_φ(r_i) = 0
            r_uphi = zeros(T, Nr)
            for i in 2:Nr
                dr = r[i] - r[i-1]
                r_uphi[i] = r_uphi[i-1] + 0.5 * (rhs[i-1] + rhs[i]) * dr
            end

            # ū_φ,ℓm = (r ū_φ) / r
            uphi_coeffs[(ℓ,m)] = r_uphi ./ r

            # Compute derivative
            duphi_dr_coeffs[(ℓ,m)] = cd.D1 * uphi_coeffs[(ℓ,m)]

            # Apply no-slip boundary conditions
            uphi_coeffs[(ℓ,m)][1] = 0.0
            uphi_coeffs[(ℓ,m)][end] = 0.0
            duphi_dr_coeffs[(ℓ,m)][1] = 0.0
            duphi_dr_coeffs[(ℓ,m)][end] = 0.0
        end
    end

    return nothing
end
