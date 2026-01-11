# =============================================================================
#  Basic State Operators for Linear Stability Analysis
#
#  Implements the linearized operators for stability analysis on an
#  axisymmetric basic state with thermal wind-balanced zonal flow.
#
#  Theory:
#  -------
#  Basic state: θ̄(r,θ) = Σ_ℓ θ̄_ℓ0(r) Y_ℓ0(θ)
#               ū_φ(r,θ) = Σ_ℓ ū_φ,ℓ0(r) Y_ℓ0(θ)
#
#  Perturbations: θ'(r,θ,φ,t) = Σ_ℓ θ'_ℓm(r,t) Y_ℓm(θ,φ)
#
#  Linearized equations add three types of terms:
#  1. Advection: (ū · ∇)θ' = (ū_φ/(r sin θ)) ∂θ'/∂φ
#  2. Shear: (u' · ∇)ū = u'_r ∂ū_φ/∂r + u'_θ ∂ū_φ/∂θ
#  3. Temperature gradient: (u' · ∇)θ̄ = u'_r ∂θ̄/∂r + u'_θ ∂θ̄/∂θ
#
#  These couple different ℓ modes through spherical harmonic products.
# =============================================================================

using LinearAlgebra
using SparseArrays
using WignerSymbols

"""
    BasicStateOperators{T<:Real}

Container for precomputed basic state linearized operators.

Fields:
- `advection_blocks::Dict{Tuple{Int,Int}, Matrix{T}}` - Advection operators A[ℓ_pert, ℓ_bs]
- `shear_blocks::Dict{Tuple{Int,Int}, Matrix{T}}` - Shear production operators
- `temp_grad_blocks::Dict{Tuple{Int,Int}, Matrix{T}}` - Temperature gradient operators
- `coupling_structure::Vector{Tuple{Int,Int}}` - List of (ℓ_pert, ℓ_bs) pairs that couple

The blocks represent coupling between perturbation mode ℓ_pert and basic state mode ℓ_bs.
"""
struct BasicStateOperators{T<:Real}
    advection_blocks::Dict{Tuple{Int,Int}, Matrix{ComplexF64}}
    shear_radial_blocks::Dict{Tuple{Int,Int}, Matrix{ComplexF64}}
    shear_theta_blocks::Dict{Tuple{Int,Int}, Matrix{ComplexF64}}
    temp_grad_radial_blocks::Dict{Tuple{Int,Int}, Matrix{ComplexF64}}
    temp_grad_theta_blocks::Dict{Tuple{Int,Int}, Matrix{ComplexF64}}
    coupling_structure::Vector{Tuple{Int,Int}}
end


"""
    compute_spherical_harmonic_coupling(ℓ_pert::Int, ℓ_bs::Int, m::Int)

Compute coupling coefficients between perturbation mode (ℓ_pert, m) and
basic state mode (ℓ_bs, 0) through spherical harmonic integrals.

For products like Y_ℓpert,m × Y_ℓbs,0, we get contributions to Y_ℓ',m where
ℓ' ranges over |ℓ_pert - ℓ_bs| to ℓ_pert + ℓ_bs with appropriate selection rules.

Returns:
- `coupling_coeffs::Dict{Int, Float64}` - Coefficients for each coupled ℓ' mode
"""
function compute_spherical_harmonic_coupling(ℓ_pert::Int, ℓ_bs::Int, m::Int)
    # For axisymmetric basic state (m_bs = 0), the product
    # Y_ℓpert,m × Y_ℓbs,0 couples to modes with same m
    #
    # Using Clebsch-Gordan coefficients and selection rules:
    # ℓ' ∈ [|ℓ_pert - ℓ_bs|, ℓ_pert + ℓ_bs] with ℓ' + ℓ_pert + ℓ_bs even

    coupling_coeffs = Dict{Int, Float64}()

    ℓ_min = abs(ℓ_pert - ℓ_bs)
    ℓ_max = ℓ_pert + ℓ_bs

    for ℓ_prime in ℓ_min:ℓ_max
        # Selection rule: ℓ' + ℓ_pert + ℓ_bs must be even
        if (ℓ_prime + ℓ_pert + ℓ_bs) % 2 != 0
            continue
        end

        # Also need ℓ_prime >= m
        if ℓ_prime < m
            continue
        end

        # Gaunt coefficient (Wigner 3j symbol related)
        # Simplified formula for m_bs = 0 case
        coeff = compute_gaunt_coefficient(ℓ_pert, m, ℓ_bs, 0, ℓ_prime, m)

        if abs(coeff) > 1e-14
            coupling_coeffs[ℓ_prime] = coeff
        end
    end

    return coupling_coeffs
end


"""
    wigner3j_000(ℓ1::Int, ℓ2::Int, ℓ3::Int)

Compute Wigner 3j symbol with all m = 0:
    ⎛ℓ1  ℓ2  ℓ3⎞
    ⎝0   0   0 ⎠

Uses WignerSymbols.jl for accurate computation.
"""
function wigner3j_000(ℓ1::Int, ℓ2::Int, ℓ3::Int)
    return Float64(WignerSymbols.wigner3j(ℓ1, ℓ2, ℓ3, 0, 0, 0))
end

"""
    compute_gaunt_coefficient(ℓ1::Int, m1::Int, ℓ2::Int, m2::Int, ℓ3::Int, m3::Int)

Compute Gaunt coefficient (integral of three spherical harmonics):

    ∫ Y_ℓ1,m1 × Y_ℓ2,m2 × conj(Y_ℓ3,m3) dΩ

Using Wigner 3j symbols:
    G = √[(2ℓ1+1)(2ℓ2+1)(2ℓ3+1)/(4π)] × ⎛ℓ1  ℓ2  ℓ3⎞ ⎛ℓ1  ℓ2   ℓ3 ⎞
                                        ⎝0   0   0 ⎠ ⎝m1  m2  -m3⎠

For axisymmetric basic state (m2 = 0), this simplifies to:
    G = √[(2ℓ1+1)(2ℓ2+1)(2ℓ3+1)/(4π)] × ⎛ℓ1  ℓ2  ℓ3⎞ ⎛ℓ1  ℓ2  ℓ3 ⎞ × δ_{m1,m3}
                                        ⎝0   0   0 ⎠ ⎝m1  0   -m1⎠
"""
function compute_gaunt_coefficient(ℓ1::Int, m1::Int, ℓ2::Int, m2::Int, ℓ3::Int, m3::Int)
    # Selection rules for ⟨Y_{ℓ1,m1} Y_{ℓ2,m2} Y_{ℓ3,m3}*⟩
    if m1 + m2 != m3
        return 0.0
    end
    if abs(m1) > ℓ1 || abs(m2) > ℓ2 || abs(m3) > ℓ3
        return 0.0
    end
    if !((abs(ℓ1 - ℓ2) <= ℓ3 <= ℓ1 + ℓ2))
        return 0.0
    end
    if (ℓ1 + ℓ2 + ℓ3) % 2 != 0
        return 0.0
    end

    w3j_1 = wigner3j_000(ℓ1, ℓ2, ℓ3)
    abs(w3j_1) < 1e-14 && return 0.0

    w3j_2 = Float64(WignerSymbols.wigner3j(ℓ1, ℓ2, ℓ3, m1, m2, -m3))
    abs(w3j_2) < 1e-14 && return 0.0

    norm_factor = sqrt((2 * ℓ1 + 1) * (2 * ℓ2 + 1) * (2 * ℓ3 + 1) / (4 * π))
    phase = isodd(m3) ? -1.0 : 1.0
    return phase * norm_factor * w3j_1 * w3j_2
end


"""
    _theta_derivative_coeff(l::Int, m::Int)

Compute spherical harmonic θ-derivative coupling coefficients.

For ∂Y_ℓm/∂θ → Y_{ℓ±1,m}, the standard recurrence relations give:
- c_plus (coupling to ℓ+1): -(ℓ+1) × √[((ℓ+1)²-m²)/((2ℓ+1)(2ℓ+3))]
- c_minus (coupling to ℓ-1): +ℓ × √[(ℓ²-m²)/((2ℓ-1)(2ℓ+1))]

These follow from the associated Legendre recurrence:
  (1-x²) dP_ℓ^m/dx = -ℓx P_ℓ^m + (ℓ+m) P_{ℓ-1}^m

Returns (c_plus, c_minus).
"""
function _theta_derivative_coeff(l::Int, m::Int)
    if l < abs(m)
        return (0.0, 0.0)
    end

    c_plus = 0.0
    c_minus = 0.0

    # Coupling to ℓ+1: coefficient = -(ℓ+1) × √[((ℓ+1)²-m²)/((2ℓ+1)(2ℓ+3))]
    if l > 0
        num_plus = (l + 1)^2 - m^2
        den_plus = (2l + 1) * (2l + 3)
        c_plus = -(l + 1) * sqrt(num_plus / den_plus)
    end

    # Coupling to ℓ-1: coefficient = +ℓ × √[(ℓ²-m²)/((2ℓ-1)(2ℓ+1))]
    if l > abs(m)
        num_minus = l^2 - m^2
        den_minus = (2l - 1) * (2l + 1)
        c_minus = l * sqrt(num_minus / den_minus)
    end

    return (c_plus, c_minus)
end

function _meridional_coupling(l_input::Int, l_bs::Int, l_output::Int, m::Int)
    c_plus, c_minus = _theta_derivative_coeff(l_bs, 0)
    coupling = 0.0

    if abs(c_plus) > 1e-14
        l_temp = l_bs + 1
        coupling += c_plus * compute_gaunt_coefficient(l_input, m, l_temp, 0, l_output, m)
    end
    if abs(c_minus) > 1e-14 && l_bs > 0
        l_temp = l_bs - 1
        coupling += c_minus * compute_gaunt_coefficient(l_input, m, l_temp, 0, l_output, m)
    end

    return coupling
end

struct AzimuthalCouplingCache
    m::Int
    weight::Float64
    y_m::Matrix{Float64}
    y_0::Matrix{Float64}
end

# Note: _double_factorial, _associated_legendre_table, and _normalization_table
# are defined in get_velocity.jl which is included before this file.

function _build_azimuthal_coupling_cache(m::Int, lmax_m::Int, lmax_0::Int)
    ntheta = max(64, 4 * max(lmax_m, lmax_0) + 1)
    k = collect(1:ntheta)
    mu = cos.((2 .* k .- 1) .* (pi / (2 * ntheta)))
    weight = pi / ntheta

    Pm = _associated_legendre_table(m, lmax_m, mu)
    P0 = _associated_legendre_table(0, lmax_0, mu)
    Nm = _normalization_table(m, lmax_m)
    N0 = _normalization_table(0, lmax_0)

    y_m = similar(Pm)
    for i in axes(Pm, 1)
        y_m[i, :] .= Nm[i] .* Pm[i, :]
    end

    y_0 = similar(P0)
    for i in axes(P0, 1)
        y_0[i, :] .= N0[i] .* P0[i, :]
    end

    return AzimuthalCouplingCache(m, weight, y_m, y_0)
end

function _azimuthal_coupling_matrix(cache::AzimuthalCouplingCache, l_bs::Int)
    y_bs = view(cache.y_0, l_bs + 1, :)
    weighted = cache.y_m .* y_bs'
    return (cache.y_m * weighted') .* (2 * pi * cache.weight)
end

"""
    build_basic_state_operators(basic_state::BasicState{T},
                                 op::LinearStabilityOperator{T},
                                 m::Int) where T

Build all linearized operators for stability analysis on a basic state.

Arguments:
- `basic_state` - The axisymmetric basic state (θ̄, ū_φ)
- `op` - Linear stability operator structure (contains radial operators)
- `m` - Azimuthal wavenumber of perturbation

Returns:
- `BasicStateOperators` - Precomputed operator blocks for all ℓ couplings

The operators are organized as blocks connecting perturbation mode ℓ_pert
to basic state mode ℓ_bs.
"""
function build_basic_state_operators(basic_state::BasicState{T},
                                      op,
                                      m::Int) where T

    # Extract radial operators
    r = collect(op.r)  # Radial collocation points
    Nr = length(r)

    # Radial differentiation operator (from Chebyshev differentiation structure)
    Dr = op.cd.D1

    # Storage for operator blocks
    # Key: (ℓ_output, ℓ_input) - coupling from input mode to output mode
    advection_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    shear_radial_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    shear_theta_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    temp_grad_radial_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    temp_grad_theta_blocks = Dict{Tuple{Int,Int}, Matrix{ComplexF64}}()
    coupling_structure = Tuple{Int,Int}[]

    # Get basic state modes
    ℓ_bs_modes = sort(collect(keys(basic_state.theta_coeffs)))

    # Get perturbation modes from operator
    perturbation_modes = sort(collect(keys(op.index_map)))
    ℓ_pert_modes = unique([ℓ for (ℓ, field) in perturbation_modes])
    ℓ_pert_set = Set(ℓ_pert_modes)
    lmax_pert = maximum(ℓ_pert_modes)
    lmax_bs = maximum(ℓ_bs_modes)
    coupling_tol = 1e-14

    azimuthal_cache = m == 0 ? nothing : _build_azimuthal_coupling_cache(m, lmax_pert, lmax_bs)

    println("Building basic state operators...")
    println("  Basic state modes (ℓ_bs): ", ℓ_bs_modes)
    println("  Perturbation modes (ℓ_pert): ", ℓ_pert_modes)
    println("  Azimuthal wavenumber m = ", m)

    # Loop over all basic state modes
    for ℓ_bs in ℓ_bs_modes
        # Get basic state coefficients
        theta_coeff = basic_state.theta_coeffs[ℓ_bs]
        uphi_coeff = basic_state.uphi_coeffs[ℓ_bs]
        duphi_dr = basic_state.duphi_dr_coeffs[ℓ_bs]
        dtheta_dr = basic_state.dtheta_dr_coeffs[ℓ_bs]

        uphi_max = maximum(abs.(uphi_coeff))
        theta_max = maximum(abs.(theta_coeff))

        # Skip if this mode is negligible
        if theta_max < 1e-14 && uphi_max < 1e-14
            continue
        end

        adv_coupling_matrix = nothing
        if azimuthal_cache !== nothing && uphi_max > coupling_tol
            adv_coupling_matrix = _azimuthal_coupling_matrix(azimuthal_cache, ℓ_bs)
        end

        # Loop over input perturbation modes
        for ℓ_input in ℓ_pert_modes
            if ℓ_input < m
                continue
            end

            # Compute spherical harmonic coupling:
            # Product Y_ℓ_input,m × Y_ℓ_bs,0 → Σ_ℓ' coupling[ℓ'] × Y_ℓ',m
            coupling_coeffs = compute_spherical_harmonic_coupling(ℓ_input, ℓ_bs, m)

            # Loop over all OUTPUT modes that receive coupling
            for (ℓ_output, coupling_coeff) in coupling_coeffs
                # Only include output modes that exist in the perturbation basis
                if !(ℓ_output in ℓ_pert_set) || ℓ_output < m
                    continue
                end

                # Record this coupling
                if !((ℓ_output, ℓ_input) in coupling_structure)
                    push!(coupling_structure, (ℓ_output, ℓ_input))
                end

                # Angular momentum quantum number for input mode
                L_input = T(ℓ_input * (ℓ_input + 1))

                # =====================================================================
                # 2. Radial shear: -u'_r × ∂ū_φ/∂r
                # =====================================================================
                if uphi_max > 1e-14
                    shear_op = -L_input * coupling_coeff * Diagonal(duphi_dr)

                    if !haskey(shear_radial_blocks, (ℓ_output, ℓ_input))
                        shear_radial_blocks[(ℓ_output, ℓ_input)] = Matrix(shear_op)
                    else
                        shear_radial_blocks[(ℓ_output, ℓ_input)] .+= Matrix(shear_op)
                    end
                end

                # =====================================================================
                # 4. Radial temperature gradient: -u'_r × ∂θ̄/∂r
                # =====================================================================
                # u'_r ~ ℓ(ℓ+1)/r² × P for poloidal potential P
                if theta_max > 1e-14
                    temp_grad_op = -L_input * coupling_coeff * Diagonal(dtheta_dr)

                    if !haskey(temp_grad_radial_blocks, (ℓ_output, ℓ_input))
                        temp_grad_radial_blocks[(ℓ_output, ℓ_input)] = Matrix(temp_grad_op)
                    else
                        temp_grad_radial_blocks[(ℓ_output, ℓ_input)] .+= Matrix(temp_grad_op)
                    end
                end

            end

            # =====================================================================
            # 1. Azimuthal advection: (ū_φ/(r sin θ)) ∂/∂φ = im·m × ū_φ/(r sin θ)
            #    The 1/sinθ factor is accounted for by the quadrature-based coupling.
            # =====================================================================
            if adv_coupling_matrix !== nothing && m != 0
                idx_in = ℓ_input - m + 1
                for ℓ_output in ℓ_pert_modes
                    if ℓ_output < m
                        continue
                    end
                    idx_out = ℓ_output - m + 1
                    adv_coupling = adv_coupling_matrix[idx_out, idx_in]
                    abs(adv_coupling) < coupling_tol && continue

                    if !((ℓ_output, ℓ_input) in coupling_structure)
                        push!(coupling_structure, (ℓ_output, ℓ_input))
                    end

                    adv_operator = im * m * adv_coupling * Diagonal(uphi_coeff ./ r)

                    if !haskey(advection_blocks, (ℓ_output, ℓ_input))
                        advection_blocks[(ℓ_output, ℓ_input)] = Matrix(adv_operator)
                    else
                        advection_blocks[(ℓ_output, ℓ_input)] .+= Matrix(adv_operator)
                    end
                end
            end

            # =====================================================================
            # 3/5. Meridional shear and temperature-gradient terms
            # =====================================================================
            if uphi_max > coupling_tol || theta_max > coupling_tol
                for ℓ_output in ℓ_pert_modes
                    if ℓ_output < m
                        continue
                    end
                    meridional_coeff = _meridional_coupling(ℓ_input, ℓ_bs, ℓ_output, m)
                    abs(meridional_coeff) < coupling_tol && continue

                    if !((ℓ_output, ℓ_input) in coupling_structure)
                        push!(coupling_structure, (ℓ_output, ℓ_input))
                    end

                    if uphi_max > coupling_tol
                        shear_theta_op = -meridional_coeff * (Diagonal(uphi_coeff) * Dr)
                        if !haskey(shear_theta_blocks, (ℓ_output, ℓ_input))
                            shear_theta_blocks[(ℓ_output, ℓ_input)] = Matrix(shear_theta_op)
                        else
                            shear_theta_blocks[(ℓ_output, ℓ_input)] .+= Matrix(shear_theta_op)
                        end
                    end

                    if theta_max > coupling_tol
                        temp_grad_theta_op = -meridional_coeff * (Diagonal(theta_coeff) * Dr)
                        if !haskey(temp_grad_theta_blocks, (ℓ_output, ℓ_input))
                            temp_grad_theta_blocks[(ℓ_output, ℓ_input)] = Matrix(temp_grad_theta_op)
                        else
                            temp_grad_theta_blocks[(ℓ_output, ℓ_input)] .+= Matrix(temp_grad_theta_op)
                        end
                    end
                end
            end
        end
    end

    println("  Built ", length(coupling_structure), " operator blocks")

    # Count non-zero blocks
    n_nonzero_adv = count(block -> maximum(abs.(block)) > 1e-14, values(advection_blocks))
    println("  Non-zero advection blocks: ", n_nonzero_adv)

    return BasicStateOperators{T}(
        advection_blocks,
        shear_radial_blocks,
        shear_theta_blocks,
        temp_grad_radial_blocks,
        temp_grad_theta_blocks,
        coupling_structure
    )
end


"""
    add_basic_state_operators!(A::Matrix, B::Matrix,
                                basic_state_ops::BasicStateOperators,
                                op::LinearStabilityOperator,
                                m::Int)

Add basic state operators to the assembled A matrix.

Modifies A in place to include:
- Advection by zonal flow
- Shear production
- Temperature gradient advection

The coupling pairs are indexed as (ℓ_output, ℓ_input), meaning the operator
couples from input mode ℓ_input to output mode ℓ_output through the basic state.

Arguments:
- `A` - Operator matrix (modified in place)
- `B` - Mass matrix (not modified, included for consistency)
- `basic_state_ops` - Precomputed basic state operators
- `op` - Linear stability operator structure
- `m` - Azimuthal wavenumber
"""
function add_basic_state_operators!(A::Matrix, B::Matrix,
                                     basic_state_ops::BasicStateOperators,
                                     op,
                                     m::Int)

    println("Adding basic state operators to A matrix...")

    # Loop over all coupling pairs (ℓ_output, ℓ_input)
    for (ℓ_output, ℓ_input) in basic_state_ops.coupling_structure

        # Get indices for output and input modes
        if !haskey(op.index_map, (ℓ_output, :P)) || !haskey(op.index_map, (ℓ_input, :P))
            continue
        end

        # Output mode indices (row indices in A)
        P_out_idx = op.index_map[(ℓ_output, :P)]
        Θ_out_idx = haskey(op.index_map, (ℓ_output, :Θ)) ? op.index_map[(ℓ_output, :Θ)] : nothing
        T_out_idx = haskey(op.index_map, (ℓ_output, :T)) ? op.index_map[(ℓ_output, :T)] : nothing

        # Input mode indices (column indices in A)
        P_in_idx = op.index_map[(ℓ_input, :P)]
        Θ_in_idx = haskey(op.index_map, (ℓ_input, :Θ)) ? op.index_map[(ℓ_input, :Θ)] : nothing
        T_in_idx = haskey(op.index_map, (ℓ_input, :T)) ? op.index_map[(ℓ_input, :T)] : nothing

        # =====================================================================
        # 1. Add advection operator to scalar fields (P, T, Θ)
        #    (ū_φ/(r sin θ)) ∂/∂φ acts as im·m × ū_φ/(r sin θ)
        # =====================================================================
        if haskey(basic_state_ops.advection_blocks, (ℓ_output, ℓ_input))
            adv_block = basic_state_ops.advection_blocks[(ℓ_output, ℓ_input)]
            if P_out_idx !== nothing && P_in_idx !== nothing
                A[P_out_idx, P_in_idx] .+= adv_block
            end
            if T_out_idx !== nothing && T_in_idx !== nothing
                A[T_out_idx, T_in_idx] .+= adv_block
            end
            if Θ_out_idx !== nothing && Θ_in_idx !== nothing
                A[Θ_out_idx, Θ_in_idx] .+= adv_block
            end
        end

        # =====================================================================
        # 2. Add radial temperature gradient term
        #    ∂θ'_ℓ_output/∂t term: -u'_r,ℓ_input × ∂θ̄/∂r
        #    This couples P_ℓ_input → θ'_ℓ_output (u_r from poloidal)
        # =====================================================================
        if Θ_out_idx !== nothing
            if haskey(basic_state_ops.temp_grad_radial_blocks, (ℓ_output, ℓ_input))
                temp_grad_block = basic_state_ops.temp_grad_radial_blocks[(ℓ_output, ℓ_input)]
                A[Θ_out_idx, P_in_idx] .+= temp_grad_block
            end
        end

        # =====================================================================
        # 2b. Add meridional temperature gradient term
        #     ∂θ'_ℓ_output/∂t term: -(u'_θ/r) × ∂θ̄/∂θ
        # =====================================================================
        if Θ_out_idx !== nothing
            if haskey(basic_state_ops.temp_grad_theta_blocks, (ℓ_output, ℓ_input))
                temp_grad_theta_block = basic_state_ops.temp_grad_theta_blocks[(ℓ_output, ℓ_input)]
                A[Θ_out_idx, P_in_idx] .+= temp_grad_theta_block
            end
        end

        # =====================================================================
        # 3. Add radial shear to toroidal equation
        #    ∂u'_φ,ℓ_output/∂t term: -u'_r,ℓ_input × ∂ū_φ/∂r
        #    This couples P_ℓ_input → T_ℓ_output
        # =====================================================================
        if T_out_idx !== nothing
            if haskey(basic_state_ops.shear_radial_blocks, (ℓ_output, ℓ_input))
                shear_block = basic_state_ops.shear_radial_blocks[(ℓ_output, ℓ_input)]
                A[T_out_idx, P_in_idx] .+= shear_block
            end
        end

        # =====================================================================
        # 3b. Add meridional shear to toroidal equation
        #     ∂u'_φ,ℓ_output/∂t term: -(u'_θ/r) × ∂ū_φ/∂θ
        # =====================================================================
        if T_out_idx !== nothing
            if haskey(basic_state_ops.shear_theta_blocks, (ℓ_output, ℓ_input))
                shear_theta_block = basic_state_ops.shear_theta_blocks[(ℓ_output, ℓ_input)]
                A[T_out_idx, P_in_idx] .+= shear_theta_block
            end
        end
    end

    println("  Basic state operators added successfully")

    return nothing
end
