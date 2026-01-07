# =============================================================================
#  Linear Stability Operator (chebyshev + KrylovKit)
#
#  This implementation uses sparse spectral methods
#  Fortran/C code: the matrices are assembled explicitly from Chebyshev
#  differentiation operators with the same r-weighted combinations and tau
#  boundary conditions (Dirichlet / stress-free).
# =============================================================================

using LinearAlgebra
using KrylovKit
using LinearMaps
using Parameters
using Random
using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM

import ..Cross: ChebyshevDiffn

const _fourπ = 4π

@inline function _symmetry_flag(sym::Symbol)
    sym === :symmetric && return 1
    sym === :antisymmetric && return -1
    sym === :both && return nothing
    error("Invalid equatorial symmetry flag $sym")
end

@inline function poloidal_tau_indices(idx::UnitRange{Int})
    length(idx) ≥ 4 || throw(ArgumentError("Need at least 4 radial points to impose boundary conditions."))
    ri = first(idx)
    ro = last(idx)
    return ri, ri + 1, ro - 1, ro
end

@inline function toroidal_boundary_indices(idx::UnitRange{Int})
    return first(idx), last(idx)
end

@inline function temperature_boundary_indices(idx::UnitRange{Int})
    return first(idx), last(idx)
end

# -----------------------------------------------------------------------------
#  Parameter Structure
# -----------------------------------------------------------------------------

@with_kw struct OnsetParams{T<:Real, BS}
    E::T
    Pr::T = one(E)
    Ra::T
    χ::T
    m::Int
    lmax::Int
    Nr::Int
    ri::T = χ
    ro::T = one(E)
    L::T = ro - ri
    mechanical_bc::Symbol = :no_slip
    thermal_bc::Symbol = :fixed_temperature
    use_sparse_weighting::Bool = true
    equatorial_symmetry::Symbol = :both
    basic_state::BS = nothing

    function OnsetParams{T,BS}(E, Pr, Ra, χ, m, lmax, Nr, ri, ro, L,
                           mechanical_bc, thermal_bc, use_sparse_weighting,
                           equatorial_symmetry,
                           basic_state::BS) where {T,BS}
        @assert 0 < χ < 1 "Radius ratio must satisfy 0 < χ < 1"
        @assert E > 0 "Ekman number must be positive"
        @assert Pr > 0 "Prandtl number must be positive"
        @assert m ≥ 0 "Azimuthal wavenumber must be non-negative"
        @assert lmax ≥ m "lmax must be ≥ m"
        @assert Nr ≥ 4 "Need at least 4 radial points"
        @assert mechanical_bc in (:no_slip, :stress_free) "Invalid mechanical BC"
        @assert thermal_bc in (:fixed_temperature, :fixed_flux) "Invalid thermal BC"
        @assert equatorial_symmetry in (:both, :symmetric, :antisymmetric) "equatorial_symmetry must be :both, :symmetric, or :antisymmetric"

        new{T,BS}(E, Pr, Ra, χ, m, lmax, Nr, ri, ro, L,
               mechanical_bc, thermal_bc, use_sparse_weighting, equatorial_symmetry,
               basic_state)
    end
end

# Backward compatibility: helper function that converts ri/ro to χ
function ShellParams(; E, Pr=one(E), Ra, m, lmax, Nr,
                     ri=nothing, ro=nothing, χ=nothing,
                     mechanical_bc=:no_slip, thermal_bc=:fixed_temperature,
                     use_sparse_weighting=true, equatorial_symmetry=:both,
                     basic_state=nothing)
    # Compute χ from ri and ro if not directly provided
    if χ === nothing
        if ri !== nothing && ro !== nothing
            computed_χ = ri / ro
        elseif ri !== nothing
            computed_χ = ri
            ro = one(E)
        else
            error("ShellParams: Must provide either χ or both ri and ro")
        end
    else
        computed_χ = χ
    end

    # Use OnsetParams constructor
    return OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=computed_χ, m=m, lmax=lmax, Nr=Nr,
                      mechanical_bc=mechanical_bc, thermal_bc=thermal_bc,
                      use_sparse_weighting=use_sparse_weighting,
                      equatorial_symmetry=equatorial_symmetry,
                      basic_state=basic_state)
end

# -----------------------------------------------------------------------------
#  Linear Stability Operator
# -----------------------------------------------------------------------------

function compute_l_sets(p::OnsetParams{T}) where {T<:Real}
    if p.equatorial_symmetry === :both
        ls = collect(p.m:p.lmax)
        return Dict(:P => ls, :T => ls, :Θ => ls)
    end

    vsymm = _symmetry_flag(p.equatorial_symmetry)
    @assert vsymm !== nothing

    signm = p.m == 0 ? 0 : 1
    lm1 = p.lmax - p.m + 1
    ll_start = p.m + 1 - signm
    ll = collect(ll_start:(ll_start + lm1 - 1))

    s = Int((vsymm + 1) ÷ 2)
    pol_start = (signm + s) % 2
    tor_start = (signm + s + 1) % 2

    pol_idxs = pol_start:2:(lm1 - 1)
    tor_idxs = tor_start:2:(lm1 - 1)

    pol_ls = [ll[k + 1] for k in pol_idxs]
    tor_ls = [ll[k + 1] for k in tor_idxs]

    return Dict(:P => pol_ls, :T => tor_ls, :Θ => pol_ls)
end

struct LinearStabilityOperator{T<:Real}
    params::OnsetParams{T}
    cd::ChebyshevDiffn{T}
    r::Vector{T}
    index_map::Dict{Tuple{Int,Symbol}, UnitRange{Int}}
    l_sets::Dict{Symbol, Vector{Int}}
    total_dof::Int
    radial_cache::Dict{Tuple{Int,Int}, Matrix{T}}
end

function LinearStabilityOperator(params::OnsetParams{T}) where {T}
    cd = ChebyshevDiffn(params.Nr, [params.ri, params.ro], 4)
    r = cd.x

    l_sets = compute_l_sets(params)
    index_map = Dict{Tuple{Int,Symbol}, UnitRange{Int}}()
    idx = 1
    for ℓ in l_sets[:P]
        index_map[(ℓ, :P)] = idx:(idx + params.Nr - 1);   idx += params.Nr
    end
    for ℓ in l_sets[:T]
        index_map[(ℓ, :T)] = idx:(idx + params.Nr - 1);   idx += params.Nr
    end
    for ℓ in l_sets[:Θ]
        index_map[(ℓ, :Θ)] = idx:(idx + params.Nr - 1);   idx += params.Nr
    end

    total_dof = idx - 1
    return LinearStabilityOperator{T}(params, cd, r, index_map, l_sets, total_dof,
                                      Dict{Tuple{Int,Int}, Matrix{T}}())
end

# -----------------------------------------------------------------------------
#  Radial helper utilities
# -----------------------------------------------------------------------------

function radial_matrix(op::LinearStabilityOperator{T}, power::Int, order::Int) where {T}
    cache = op.radial_cache
    key = (power, order)
    if haskey(cache, key)
        return cache[key]
    end

    diag = Diagonal(op.r .^ power)
    mat = if order == 0
        Matrix(diag)
    elseif order == 1
        Matrix(diag * op.cd.D1)
    elseif order == 2
        Matrix(diag * op.cd.D2)
    elseif order == 3
        @assert op.cd.D3 !== nothing "Third-order derivative matrix required; increase Chebyshev order."
        Matrix(diag * op.cd.D3)
    elseif order == 4
        @assert op.cd.D4 !== nothing "Fourth-order derivative matrix required; increase Chebyshev order."
        Matrix(diag * op.cd.D4)
    else
        throw(ArgumentError("Unsupported derivative order $order"))
    end

    cache[key] = mat
    return mat
end

function impose_boundary_conditions!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                                     op::LinearStabilityOperator{T}) where {T<:Real}
    p = op.params
    D1 = op.cd.D1
    D2 = op.cd.D2

    for ℓ in op.l_sets[:P]
        P_idx = op.index_map[(ℓ, :P)]
        ri, inner_tau, outer_tau, ro = poloidal_tau_indices(P_idx)

        # Dirichlet: P = 0
        A[ri, :] .= 0;   B[ri, :] .= 0;   A[ri, ri] = 1
        A[ro, :] .= 0;   B[ro, :] .= 0;   A[ro, ro] = 1

        if p.mechanical_bc == :no_slip
            A[inner_tau, :] .= 0; B[inner_tau, :] .= 0; A[inner_tau, P_idx] .= D1[1, :]
            A[outer_tau, :] .= 0; B[outer_tau, :] .= 0; A[outer_tau, P_idx] .= D1[end, :]
        else
            A[inner_tau, :] .= 0; B[inner_tau, :] .= 0; A[inner_tau, P_idx] .= op.r[1] .* D2[1, :]
            A[outer_tau, :] .= 0; B[outer_tau, :] .= 0; A[outer_tau, P_idx] .= op.r[end] .* D2[end, :]
        end
    end

    for ℓ in op.l_sets[:T]
        T_idx = op.index_map[(ℓ, :T)]
        riT, roT = toroidal_boundary_indices(T_idx)
        if p.mechanical_bc == :no_slip
            A[riT, :] .= 0; B[riT, :] .= 0; A[riT, riT] = 1
            A[roT, :] .= 0; B[roT, :] .= 0; A[roT, roT] = 1
        else
            A[riT, :] .= 0; B[riT, :] .= 0
            A[roT, :] .= 0; B[roT, :] .= 0
            A[riT, T_idx] .= (-op.r[1]) .* D1[1, :]
            A[roT, T_idx] .= (-op.r[end]) .* D1[end, :]
            A[riT, riT] += 1
            A[roT, roT] += 1
        end
    end

    for ℓ in op.l_sets[:Θ]
        Θ_idx = op.index_map[(ℓ, :Θ)]
        riΘ, roΘ = temperature_boundary_indices(Θ_idx)
        if p.thermal_bc == :fixed_temperature
            A[riΘ, :] .= 0; B[riΘ, :] .= 0; A[riΘ, riΘ] = 1
            A[roΘ, :] .= 0; B[roΘ, :] .= 0; A[roΘ, roΘ] = 1
        else
            A[riΘ, :] .= 0; B[riΘ, :] .= 0; A[riΘ, Θ_idx] .= D1[1, :]
            A[roΘ, :] .= 0; B[roΘ, :] .= 0; A[roΘ, Θ_idx] .= D1[end, :]
        end
    end
end

# -----------------------------------------------------------------------------
#  Matrix Assembly
# -----------------------------------------------------------------------------

function assemble_matrices(op::LinearStabilityOperator{T}) where {T<:Real}
    p = op.params
    n = op.total_dof
    A = zeros(Complex{T}, n, n)
    B = zeros(Complex{T}, n, n)

    Ek = p.E
    Pr = p.Pr
    m = p.m
    ri = p.ri
    ro = p.ro
    gap = ro - ri

    # Convert gap-based Rayleigh number to internal Ra as per Kore's approach:
    # Ra_internal = Ra_gap / gap^3  (where gap = ro - ri = 1 - χ when ro = 1)
    # This is necessary because the non-dimensionalization uses r_o as the length scale
    # but Ra_gap is defined using the gap L = r_o - r_i as the length scale.
    # Reference: Barik et al. (2023), Onset of convection paper, Kore parameters.py
    Ra_internal = p.Ra / gap^3
    beyonce = -Ra_internal * Ek^2 / Pr
    thermaD = Ek / Pr

    R0 = radial_matrix(op, 0, 0)
    R1D0 = radial_matrix(op, 1, 0)
    R1D1 = radial_matrix(op, 1, 1)
    R2D0 = radial_matrix(op, 2, 0)
    R2D1 = radial_matrix(op, 2, 1)
    R2D2 = radial_matrix(op, 2, 2)
    R3D0 = radial_matrix(op, 3, 0)
    R3D1 = radial_matrix(op, 3, 1)
    R3D2 = radial_matrix(op, 3, 2)
    R3D3 = radial_matrix(op, 3, 3)
    R4D0 = radial_matrix(op, 4, 0)
    R4D1 = radial_matrix(op, 4, 1)
    R4D2 = radial_matrix(op, 4, 2)
    R4D4 = radial_matrix(op, 4, 4)

    TT = eltype(op.r)
    poloidal_ls = op.l_sets[:P]
    toroidal_ls = op.l_sets[:T]
    for ℓ in poloidal_ls
        L = TT(ℓ * (ℓ + 1))

        P_idx = op.index_map[(ℓ, :P)]
        Θ_idx = haskey(op.index_map, (ℓ, :Θ)) ? op.index_map[(ℓ, :Θ)] : nothing

        B[P_idx, P_idx] = -Complex.(L * (L * R2D0 - 2 * R3D1 - R4D2))

        # Poloidal diagonal
        coriolis_p = 2im * m * (-L * R2D0 + 2 * R3D1 + R4D2)
        viscous_p = Ek * L * (-L*(ℓ+2)*(ℓ-1) * R0 + 2*L * R2D2 - 4 * R3D3 - R4D4)
        A[P_idx, P_idx] .+= Complex.(coriolis_p - viscous_p)

        # Poloidal ↔ toroidal coupling
        if haskey(op.index_map, (ℓ - 1, :T))
            Cminus = (ℓ^2 - 1) * sqrt(max(zero(TT), ℓ^2 - m^2)) / (2ℓ - 1)
            coupling = 2 * Cminus * ((ℓ - 1) * R3D0 - R4D1)
            A[P_idx, op.index_map[(ℓ-1, :T)]] .+= Complex.(coupling)
        end
        if haskey(op.index_map, (ℓ + 1, :T))
            Cplus = ℓ*(ℓ+2) * sqrt(max(zero(TT), (ℓ+m+1)*(ℓ-m+1))) / (2ℓ + 3)
            coupling = 2 * Cplus * (-(ℓ + 2) * R3D0 - R4D1)
            A[P_idx, op.index_map[(ℓ+1, :T)]] .+= Complex.(coupling)
        end

        # Temperature equation blocks for matching Θ ℓ
        if Θ_idx !== nothing
            if p.use_sparse_weighting
                B_theta = R3D0
                adv_coeff = ri / gap
                adv_matrix = R0
                diffusion = -L * R1D0 + 2 * R2D1 + R3D2
            else
                B_theta = R2D0
                adv_coeff = one(TT)
                adv_matrix = R2D0
                diffusion = -L * R0 + 2 * R1D1 + R2D2
            end

            B[Θ_idx, Θ_idx] = Complex.(B_theta)

            # Temperature gradient coupling: only add if NO basic state
            # (basic state will provide explicit gradient through basic_state_operators)
            if p.basic_state === nothing
                thermal_adv = (L * adv_coeff) .* adv_matrix
                A[Θ_idx, P_idx] .+= Complex.(thermal_adv)
            end

            A[Θ_idx, Θ_idx] .+= Complex.(thermaD * diffusion)

            # Buoyancy coupling: temperature → velocity (OFF-DIAGONAL)
            # This term is ALWAYS present regardless of basic state
            buoyancy = beyonce * L * R4D0
            A[P_idx, Θ_idx] .+= Complex.(buoyancy)
        end
    end

    # Add basic state operators if present (axisymmetric zonal flow stability)
    if p.basic_state !== nothing
        println("Adding basic state operators for axisymmetric zonal flow...")
        bs_ops = build_basic_state_operators(p.basic_state, op, p.m)
        add_basic_state_operators!(A, B, bs_ops, op, p.m)
    end

    for ℓ in toroidal_ls
        L = TT(ℓ * (ℓ + 1))
        T_idx = op.index_map[(ℓ, :T)]

        B[T_idx, T_idx] = -Complex.(L * R2D0)

        # Toroidal diagonal
        # BUG FIX 2025-10-27: Removed incorrect L factor from Coriolis term
        # Kore operators.py:121-122: out = -2j*par.m*r2_D0_v (NO L factor!)
        # The toroidal Coriolis term should NOT have the L factor
        coriolis_t = -2im * m * R2D0
        viscous_t = Ek * L * (-L * R0 + 2 * R1D1 + R2D2)
        A[T_idx, T_idx] .+= Complex.(coriolis_t - viscous_t)

        # Toroidal ↔ poloidal coupling
        if haskey(op.index_map, (ℓ - 1, :P))
            Cminus = (ℓ^2 - 1) * sqrt(max(zero(TT), ℓ^2 - m^2)) / (2ℓ - 1)
            coupling = 2 * Cminus * ((ℓ - 1) * R1D0 - R2D1)
            A[T_idx, op.index_map[(ℓ-1, :P)]] .+= Complex.(coupling)
        end
        if haskey(op.index_map, (ℓ + 1, :P))
            Cplus = ℓ*(ℓ+2) * sqrt(max(zero(TT), (ℓ+m+1)*(ℓ-m+1))) / (2ℓ + 3)
            coupling = 2 * Cplus * (-(ℓ + 2) * R1D0 - R2D1)
            A[T_idx, op.index_map[(ℓ+1, :P)]] .+= Complex.(coupling)
        end
    end

    impose_boundary_conditions!(A, B, op)

    boundary_dofs = Int[]
    for ℓ in op.l_sets[:P]
        P_idx = op.index_map[(ℓ, :P)]
        ri, inner_tau, outer_tau, ro = poloidal_tau_indices(P_idx)
        append!(boundary_dofs, (ri, inner_tau, outer_tau, ro))
    end
    for ℓ in op.l_sets[:T]
        T_idx = op.index_map[(ℓ, :T)]
        riT, roT = toroidal_boundary_indices(T_idx)
        append!(boundary_dofs, (riT, roT))
    end
    for ℓ in op.l_sets[:Θ]
        Θ_idx = op.index_map[(ℓ, :Θ)]
        riΘ, roΘ = temperature_boundary_indices(Θ_idx)
        append!(boundary_dofs, (riΘ, roΘ))
    end
    sort!(boundary_dofs)
    boundary_dofs = unique(boundary_dofs)
    interior_dofs = setdiff(collect(1:n), boundary_dofs)

    return A, B, interior_dofs, boundary_dofs
end

# -----------------------------------------------------------------------------
#  Eigenvalue solve
# -----------------------------------------------------------------------------

struct ShiftInvertMap{TF,TB,Ttmp}
    lu::TF
    B::TB
    tmp::Ttmp
end

(M::ShiftInvertMap)(y, x) = (mul!(M.tmp, M.B, x); ldiv!(y, M.lu, M.tmp); y)

function solve_eigenvalue_problem(op::LinearStabilityOperator{T};
                                  nev::Int=6,
                                  tol::Float64=1e-10,
                                  maxiter::Int=1000,
                                  which::Symbol=:LR,
                                  sigma::Union{Nothing,Number}=nothing) where {T<:Float64}

    A_full, B_full, interior_dofs, _ = assemble_matrices(op)

    # Use KrylovKit shift-invert with optimized shift
    # Default shift for onset problems: small positive real value (near Re(λ)=0)
    return _krylov_eigensolve_optimized(A_full, B_full, interior_dofs;
                                        nev=nev, tol=tol, maxiter=maxiter,
                                        which=which, sigma=sigma)
end


function find_growth_rate(op::LinearStabilityOperator; kwargs...)
    eigenvalues, eigenvectors, info = solve_eigenvalue_problem(op; kwargs...)
    idx = argmax(real.(eigenvalues))
    λ = eigenvalues[idx]
    σ = real(λ)
    ω = imag(λ)
    return σ, ω, eigenvectors[idx]
end

# Backward compatibility wrapper
function leading_modes(params::OnsetParams; nθ=nothing, kwargs...)
    # nθ is ignored for now - it's not used in the current eigenvalue solver
    # but kept for API compatibility
    op = LinearStabilityOperator(params)
    eigenvalues, eigenvectors, info = solve_eigenvalue_problem(op; kwargs...)
    return eigenvalues, eigenvectors, op, info
end

function _krylov_eigensolve_optimized(A_full::Matrix{Complex{T}},
                                       B_full::Matrix{Complex{T}},
                                       interior_dofs::Vector{Int};
                                       nev::Int,
                                       tol::Float64,
                                       maxiter::Int,
                                       which::Symbol,
                                       sigma::Union{Nothing,Number}=nothing) where {T<:Float64}
    """
    OPTIMIZED shift-invert solver using KrylovKit.
    Automatically selects shift based on 'which' parameter for onset problems.
    """
    A = Matrix(A_full[interior_dofs, interior_dofs])
    B = Matrix(B_full[interior_dofs, interior_dofs])

    # Smart shift selection based on what eigenvalues we're targeting
    if isnothing(sigma)
        if which == :LR
            # For onset problems: target eigenvalues with largest real part
            # Critical Rayleigh number is when Re(λ) ≈ 0
            # Use small positive shift to find eigenvalues near onset
            σ = Complex{T}(10.0, 0.0)  # Much better than 0.1+0.1i!
        elseif which == :LI
            # Target largest imaginary part
            σ = Complex{T}(0.0, 10.0)
        else
            # Default: near origin
            σ = Complex{T}(1.0, 0.0)
        end
    else
        σ = Complex{T}(sigma)
    end

    lu_factor = lu(A - σ * B)
    tmp = zeros(Complex{T}, size(A, 1))
    shift_map = LinearMap{Complex{T}}(ShiftInvertMap(lu_factor, B, tmp),
                                      size(A, 1); ismutating=true)

    krylovdim = min(size(A, 1), max(nev * 8, 60))
    x0 = randn(Complex{T}, size(A, 1))

    vals_inv, vecs_int, info = eigsolve(shift_map,
                                        x0, nev, :LM;
                                        tol=tol,
                                        maxiter=maxiter,
                                        krylovdim=krylovdim,
                                        verbosity=0)

    finite = [abs(λ) > eps(T) for λ in vals_inv]
    any(finite) || error("No finite eigenvalues returned by eigensolver")
    vals_inv = vals_inv[finite]
    vecs_int = vecs_int[finite]

    eigenvalues = Complex{T}[σ + inv(λ) for λ in vals_inv]

    vecs_full = Vector{Vector{Complex{T}}}(undef, length(vecs_int))
    n_full = size(A_full, 1)
    for (i, v_int) in enumerate(vecs_int)
        full_vec = zeros(Complex{T}, n_full)
        full_vec[interior_dofs] = v_int
        vecs_full[i] = full_vec
    end

    ordering = which == :LR ? sortperm(real.(eigenvalues); rev=true) :
               which == :LM ? sortperm(abs.(eigenvalues); rev=true) :
               collect(1:length(eigenvalues))

    return eigenvalues[ordering], vecs_full[ordering], info
end

function _arnoldi_eigensolve(A_full::Matrix{Complex{T}},
                              B_full::Matrix{Complex{T}},
                              interior_dofs::Vector{Int};
                              nev::Int,
                              tol::Float64,
                              maxiter::Int,
                              which::Symbol) where {T<:Float64}
    """
    Fast eigenvalue solver using ArnoldiMethod.
    Directly targets eigenvalues by real part (:LR) without shift-invert.
    MUCH faster for onset of convection problems.
    """
    A = Matrix(A_full[interior_dofs, interior_dofs])
    B = Matrix(B_full[interior_dofs, interior_dofs])

    # Solve generalized eigenvalue problem A*v = λ*B*v
    # Convert to standard form: B^{-1}*A*v = λ*v
    B_lu = lu(B)
    inv_B_A = B_lu \ A

    # Map which symbol to ArnoldiMethod selector
    selector = if which == :LR
        LR()  # Largest Real part - optimal for onset!
    elseif which == :LI
        LI()  # Largest Imaginary part
    elseif which == :LM
        LM()  # Largest Magnitude
    else
        LR()  # Default to LR for onset problems
    end

    # Use partialschur for robust convergence
    # For onset problems, eigenvalues can be large - use more Krylov vectors
    # ArnoldiMethod uses mindim/maxdim for Krylov subspace dimension
    mindim = max(nev + 5, 10)
    maxdim = min(size(inv_B_A, 1), max(2*nev + 20, 30))
    schur_decomp, history = partialschur(inv_B_A, nev=nev, tol=tol,
                                          restarts=maxiter, which=selector,
                                          mindim=mindim, maxdim=maxdim)

    # Extract eigenvalues and eigenvectors
    eigenvalues_int, eigenvectors_int = partialeigen(schur_decomp)

    # Convert to full vectors
    vecs_full = Vector{Vector{Complex{T}}}(undef, length(eigenvectors_int))
    n_full = size(A_full, 1)
    for (i, v_int) in enumerate(eigenvectors_int)
        full_vec = zeros(Complex{T}, n_full)
        full_vec[interior_dofs] = v_int
        vecs_full[i] = full_vec
    end

    # Create info dict for compatibility
    # ArnoldiMethod.History fields: mvproducts, nconverged, converged, nev
    info = Dict("converged" => history.converged,
                "nconverged" => history.nconverged,
                "mvproducts" => history.mvproducts,
                "nev" => history.nev,
                "solver" => :arnoldi)

    return eigenvalues_int, vecs_full, info
end

function _krylov_eigensolve(A_full::Matrix{Complex{T}},
                            B_full::Matrix{Complex{T}},
                            interior_dofs::Vector{Int};
                            nev::Int,
                            tol::Float64,
                            maxiter::Int,
                            which::Symbol) where {T<:Float64}
    """
    DEPRECATED: Slow shift-invert solver using KrylovKit.
    Use _arnoldi_eigensolve instead for better performance.
    """
    A = Matrix(A_full[interior_dofs, interior_dofs])
    B = Matrix(B_full[interior_dofs, interior_dofs])

    σ = Complex{T}(0.1, 0.1)

    lu_factor = lu(A - σ * B)
    tmp = zeros(Complex{T}, size(A, 1))
    shift_map = LinearMap{Complex{T}}(ShiftInvertMap(lu_factor, B, tmp),
                                      size(A, 1); ismutating=true)

    krylovdim = min(size(A, 1), max(nev * 8, 60))
    x0 = randn(Complex{T}, size(A, 1))

    vals_inv, vecs_int, info = eigsolve(shift_map,
                                        x0, nev, :LM;
                                        tol=tol,
                                        maxiter=maxiter,
                                        krylovdim=krylovdim,
                                        verbosity=0)

    finite = [abs(λ) > eps(T) for λ in vals_inv]
    any(finite) || error("No finite eigenvalues returned by eigensolver")
    vals_inv = vals_inv[finite]
    vecs_int = vecs_int[finite]

    eigenvalues = Complex{T}[σ + inv(λ) for λ in vals_inv]

    vecs_full = Vector{Vector{Complex{T}}}(undef, length(vecs_int))
    n_full = size(A_full, 1)
    for (i, v_int) in enumerate(vecs_int)
        full_vec = zeros(Complex{T}, n_full)
        full_vec[interior_dofs] = v_int
        vecs_full[i] = full_vec
    end

    ordering = which == :LR ? sortperm(real.(eigenvalues); rev=true) :
               which == :LM ? sortperm(abs.(eigenvalues); rev=true) :
               collect(1:length(eigenvalues))

    return eigenvalues[ordering], vecs_full[ordering], info
end

function find_critical_rayleigh(E::T, Pr::T, χ::T, m::Int, lmax::Int, Nr::Int;
                                Ra_guess::T=one(T)*1e6,
                                tol::T=1e-6,
                                Ra_bracket::Tuple{T,T}=(Ra_guess/10, Ra_guess*10),
                                kwargs...) where {T<:Real}
    m_int = Int(m)
    lmax ≥ m_int || error("lmax must be ≥ m (got lmax=$lmax, m=$m_int)")

    onset_fields = Set(fieldnames(OnsetParams{T}))
    param_pairs = Pair{Symbol,Any}[]
    solver_pairs = Pair{Symbol,Any}[]
    for (key, val) in kwargs
        if key in onset_fields
            push!(param_pairs, key => val)
        else
            push!(solver_pairs, key => val)
        end
    end
    onset_kwargs = isempty(param_pairs) ? NamedTuple() : (; param_pairs...)
    solver_kwargs = isempty(solver_pairs) ? NamedTuple() : (; solver_pairs...)

    function build_operator(Ra_val::T)
        params = OnsetParams(E=E, Pr=Pr, Ra=Ra_val, χ=χ, m=m_int, lmax=lmax, Nr=Nr; onset_kwargs...)
        return LinearStabilityOperator(params)
    end

    function growth_rate_at_Ra(Ra)
        op = build_operator(Ra)
        σ, _, _ = find_growth_rate(op; solver_kwargs...)
        return σ
    end

    cache = Dict{Float64,T}()
    function sigma_cached(Ra_val::T)
        key = Float64(Ra_val)
        return get!(cache, key) do
            growth_rate_at_Ra(Ra_val)
        end
    end

    pos = Ref{Union{Nothing,Tuple{T,T}}}(nothing)
    neg = Ref{Union{Nothing,Tuple{T,T}}}(nothing)

    function add_sample!(Ra_val::T)
        σ_val = sigma_cached(Ra_val)
        if abs(σ_val) < tol
            return (:root, Ra_val)
        elseif σ_val > 0
            pos[] = (Ra_val, σ_val)
        else
            neg[] = (Ra_val, σ_val)
        end
        return (:continue, Ra_val)
    end

    # Seed with guess and bracket endpoints
    Ra_guess_T = convert(T, Ra_guess)
    state, Ra_root = add_sample!(Ra_guess_T)
    if state == :root
        op_c = build_operator(Ra_root)
        σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)
        return Ra_root, ω_c, vec_c
    end

    Ra_low = convert(T, Ra_bracket[1])
    Ra_high = convert(T, Ra_bracket[2])
    state, Ra_root = add_sample!(Ra_low)
    if state == :root
        op_c = build_operator(Ra_root)
        σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)
        return Ra_root, ω_c, vec_c
    end
    state, Ra_root = add_sample!(Ra_high)
    if state == :root
        op_c = build_operator(Ra_root)
        σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)
        return Ra_root, ω_c, vec_c
    end

    attempt = 0
    while (pos[] === nothing || neg[] === nothing) && attempt < 12
        if pos[] === nothing
            Ra_high *= T(2)
            state, Ra_root = add_sample!(Ra_high)
            if state == :root
                op_c = build_operator(Ra_root)
                σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)
                return Ra_root, ω_c, vec_c
            end
        end
        if neg[] === nothing
            Ra_low /= T(2)
            state, Ra_root = add_sample!(Ra_low)
            if state == :root
                op_c = build_operator(Ra_root)
                σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)
                return Ra_root, ω_c, vec_c
            end
        end
        attempt += 1
    end

    if pos[] === nothing || neg[] === nothing
        log_guess = log10(Float64(Ra_guess_T))
        step = 0.25
        for k in 1:80
            Ra_up = convert(T, 10.0^(log_guess + k * step))
            state, Ra_root = add_sample!(Ra_up)
            if state == :root
                op_c = build_operator(Ra_root)
                σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)
                return Ra_root, ω_c, vec_c
            end
            if pos[] !== nothing && neg[] !== nothing
                break
            end
            Ra_down = convert(T, 10.0^(log_guess - k * step))
            state, Ra_root = add_sample!(Ra_down)
            if state == :root
                op_c = build_operator(Ra_root)
                σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)
                return Ra_root, ω_c, vec_c
            end
            if pos[] !== nothing && neg[] !== nothing
                break
            end
        end
    end

    (pos[] === nothing || neg[] === nothing) && error("Could not bracket the critical Rayleigh number")

    (Ra_pos, σ_pos) = pos[]
    (Ra_neg, σ_neg) = neg[]
    if σ_pos < 0
        Ra_pos, σ_pos, Ra_neg, σ_neg = Ra_neg, σ_neg, Ra_pos, σ_pos
    end

    a = Ra_neg
    fa = σ_neg
    b = Ra_pos
    fb = σ_pos
    c = a
    fc = fa
    d = b - a
    e = d

    for _ in 1:200
        if fb == zero(fb)
            a, fa = b, fb
            break
        end
        if sign(fa) == sign(fb)
            a, fa = c, fc
            d = b - a
            e = d
        end
        if abs(fa) < abs(fb)
            c, fc = b, fb
            b, fb = a, fa
            a, fa = c, fc
        end
        tol_act = 2 * eps(T) * abs(b) + tol / 2
        mid = (a - b) / 2
        if abs(mid) <= tol_act || fb == zero(fb)
            break
        end
        if abs(e) >= tol_act && abs(fc) > abs(fb)
            s = fb / fc
            if a == c
                p = 2 * mid * s
                q = 1 - s
            else
                q = fc / fa
                r = fb / fa
                p = s * (2 * mid * q * (q - r) - (b - c) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            end
            if p > 0
                q = -q
            else
                p = -p
            end
            if 2 * p < 3 * mid * q - abs(tol_act * q) && p < abs(e * q / 2)
                e = d
                d = p / q
            else
                d = mid
                e = mid
            end
        else
            d = mid
            e = mid
        end
        c, fc = b, fb
        if abs(d) > tol_act
            b += d
        else
            b += sign(mid) * tol_act
        end
        fb = sigma_cached(b)
    end

    Ra_c = b
    op_c = build_operator(Ra_c)
    σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)

    return Ra_c, ω_c, vec_c
end

export OnsetParams, LinearStabilityOperator
export assemble_matrices
export solve_eigenvalue_problem, find_growth_rate, find_critical_rayleigh
