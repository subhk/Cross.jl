# =============================================================================
#  Kore Linear Stability Operator (chebyshev + KrylovKit)
#
#  This rewrite mirrors the algebraic formulation used in the original Kore
#  Fortran/C code: the matrices are assembled explicitly from Chebyshev
#  differentiation operators with the same r-weighted combinations and tau
#  boundary conditions (Dirichlet / stress-free) that Kore enforces.
# =============================================================================

using LinearAlgebra
using KrylovKit
using LinearMaps
using Parameters
using NonlinearSolve
using Random

import ..Cross: ChebyshevDiffn

const _fourπ = 4π

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

@with_kw struct OnsetParams{T<:Real}
    E::T
    Pr::T = one(T)
    Ra::T
    χ::T
    m::Int
    lmax::Int
    Nr::Int
    ri::T = χ
    ro::T = one(T)
    L::T = ro - ri
    mechanical_bc::Symbol = :no_slip
    thermal_bc::Symbol = :fixed_temperature
    use_kore_weighting::Bool = false
    basic_state::Nothing = nothing

    function OnsetParams{T}(E, Pr, Ra, χ, m, lmax, Nr, ri, ro, L,
                           mechanical_bc, thermal_bc, use_kore_weighting,
                           basic_state) where {T}
        @assert 0 < χ < 1 "Radius ratio must satisfy 0 < χ < 1"
        @assert E > 0 "Ekman number must be positive"
        @assert Pr > 0 "Prandtl number must be positive"
        @assert m ≥ 0 "Azimuthal wavenumber must be non-negative"
        @assert lmax ≥ m "lmax must be ≥ m"
        @assert Nr ≥ 4 "Need at least 4 radial points"
        @assert mechanical_bc in (:no_slip, :stress_free) "Invalid mechanical BC"
        @assert thermal_bc in (:fixed_temperature, :fixed_flux) "Invalid thermal BC"

        new{T}(E, Pr, Ra, χ, m, lmax, Nr, ri, ro, L,
               mechanical_bc, thermal_bc, use_kore_weighting, basic_state)
    end
end

# -----------------------------------------------------------------------------
#  Linear Stability Operator
# -----------------------------------------------------------------------------

struct LinearStabilityOperator{T<:Real}
    params::OnsetParams{T}
    cd::ChebyshevDiffn{T}
    r::Vector{T}
    index_map::Dict{Tuple{Int,Symbol}, UnitRange{Int}}
    total_dof::Int
    radial_cache::Dict{Tuple{Int,Int}, Matrix{T}}
end

function LinearStabilityOperator(params::OnsetParams{T}) where {T}
    cd = ChebyshevDiffn(params.Nr, [params.ri, params.ro], 4)
    r = cd.x

    index_map = Dict{Tuple{Int,Symbol}, UnitRange{Int}}()
    idx = 1
    for ℓ in params.m:params.lmax
        index_map[(ℓ, :P)] = idx:(idx + params.Nr - 1);   idx += params.Nr
        index_map[(ℓ, :T)] = idx:(idx + params.Nr - 1);   idx += params.Nr
        index_map[(ℓ, :Θ)] = idx:(idx + params.Nr - 1);   idx += params.Nr
    end

    total_dof = idx - 1
    return LinearStabilityOperator{T}(params, cd, r, index_map, total_dof,
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

    for ℓ in p.m:p.lmax
        P_idx = op.index_map[(ℓ, :P)]
        T_idx = op.index_map[(ℓ, :T)]
        Θ_idx = op.index_map[(ℓ, :Θ)]

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

    beyonce = -p.Ra * Ek^2 / Pr
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

    for ℓ in p.m:p.lmax
        L = T(ℓ * (ℓ + 1))

        P_idx = op.index_map[(ℓ, :P)]
        T_idx = op.index_map[(ℓ, :T)]
        Θ_idx = op.index_map[(ℓ, :Θ)]

        # B matrix blocks
        B[P_idx, P_idx] = -Complex.(L * (L * R2D0 - 2 * R3D1 - R4D2))
        B[T_idx, T_idx] = -Complex.(L * R2D0)
        if p.use_kore_weighting
            B[Θ_idx, Θ_idx] = Complex.(radial_matrix(op, 3, 0))
        else
            B[Θ_idx, Θ_idx] = Complex.(radial_matrix(op, 2, 0))
        end

        # Poloidal diagonal
        coriolis_p = 2im * m * (-L * R2D0 + 2 * R3D1 + R4D2)
        viscous_p = Ek * L * (-L*(ℓ+2)*(ℓ-1) * R0 + 2*L * R2D2 - 4 * R3D3 - R4D4)
        buoyancy = beyonce * L * R4D0
        A[P_idx, P_idx] .+= Complex.(coriolis_p - viscous_p + buoyancy)

        # Poloidal ↔ toroidal coupling
        if ℓ > p.m
            Cminus = (ℓ^2 - 1) * sqrt(max(T(0), ℓ^2 - m^2)) / (2ℓ - 1)
            coupling = 2 * Cminus * ((ℓ - 1) * R3D0 - R4D1)
            A[P_idx, op.index_map[(ℓ-1, :T)]] .+= Complex.(coupling)
        end
        if ℓ < p.lmax
            Cplus = ℓ*(ℓ+2) * sqrt(max(T(0), (ℓ+m+1)*(ℓ-m+1))) / (2ℓ + 3)
            coupling = 2 * Cplus * (-(ℓ + 2) * R3D0 - R4D1)
            A[P_idx, op.index_map[(ℓ+1, :T)]] .+= Complex.(coupling)
        end

        # Toroidal diagonal
        coriolis_t = -2im * m * L * R2D0
        viscous_t = Ek * L * (-L * R0 + 2 * R1D1 + R2D2)
        A[T_idx, T_idx] .+= Complex.(coriolis_t - viscous_t)

        # Toroidal ↔ poloidal coupling
        if ℓ > p.m
            Cminus = (ℓ^2 - 1) * sqrt(max(T(0), ℓ^2 - m^2)) / (2ℓ - 1)
            coupling = 2 * Cminus * ((ℓ - 1) * R1D0 - R2D1)
            A[T_idx, op.index_map[(ℓ-1, :P)]] .+= Complex.(coupling)
        end
        if ℓ < p.lmax
            Cplus = ℓ*(ℓ+2) * sqrt(max(T(0), (ℓ+m+1)*(ℓ-m+1))) / (2ℓ + 3)
            coupling = 2 * Cplus * (-(ℓ + 2) * R1D0 - R2D1)
            A[T_idx, op.index_map[(ℓ+1, :P)]] .+= Complex.(coupling)
        end

        # Temperature equation
        if p.use_kore_weighting
            adv_factor = ri / gap
            A[Θ_idx, P_idx] .+= Complex.(L * adv_factor * R0)
            diffusion = -L * R1D0 + 2 * R2D1 + R3D2
            A[Θ_idx, Θ_idx] .+= Complex.(thermaD * diffusion)
        else
            adv_factor = (ri * ro) / p.L
            A[Θ_idx, P_idx] .+= Complex.(L * adv_factor * radial_matrix(op, -2, 0))
            diffusion = -L * R0 + 2 * R1D1 + R2D2
            A[Θ_idx, Θ_idx] .+= Complex.(thermaD * diffusion)
        end
    end

    impose_boundary_conditions!(A, B, op)

    boundary_dofs = Int[]
    for ℓ in p.m:p.lmax
        P_idx = op.index_map[(ℓ, :P)]
        ri, inner_tau, outer_tau, ro = poloidal_tau_indices(P_idx)
        append!(boundary_dofs, (ri, inner_tau, outer_tau, ro))

        T_idx = op.index_map[(ℓ, :T)]
        riT, roT = toroidal_boundary_indices(T_idx)
        append!(boundary_dofs, (riT, roT))

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
                                  tol::Real=1e-10,
                                  maxiter::Int=1000,
                                  which::Symbol=:LR) where {T<:Real}
    A, B, _, _ = assemble_matrices(op)
    σ = Complex{T}(0)

    lu_factor = lu(A - σ * B)
    tmp = zeros(Complex{T}, size(A, 1))
    shift_map = LinearMap{Complex{T}}(ShiftInvertMap(lu_factor, B, tmp), size(A, 1); ismutating=true)

    krylovdim = min(size(A, 1), max(nev * 8, 60))
    x0 = randn(Complex{T}, size(A, 1))

    vals_inv, vecs, info = eigsolve(shift_map, x0, nev, :LM;
                                    tol=tol, maxiter=maxiter,
                                    krylovdim=krylovdim, verbosity=0)

    eigenvalues = Complex{T}[σ + inv(λ) for λ in vals_inv]

    ordering = which == :LR ? sortperm(real.(eigenvalues); rev=true) :
               which == :LM ? sortperm(abs.(eigenvalues); rev=true) :
               collect(1:length(eigenvalues))

    return eigenvalues[ordering], vecs[ordering], info
end

function find_growth_rate(op::LinearStabilityOperator; kwargs...)
    eigenvalues, eigenvectors, info = solve_eigenvalue_problem(op; kwargs...)
    idx = argmax(real.(eigenvalues))
    λ = eigenvalues[idx]
    σ = real(λ)
    ω = imag(λ)
    return σ, ω, eigenvectors[idx]
end

function find_critical_rayleigh(E::T, Pr::T, χ::T, m::Int, lmax::Int, Nr::Int;
                                Ra_guess::T=one(T)*1e6,
                                tol::T=1e-6,
                                Ra_bracket::Tuple{T,T}=(Ra_guess/10, Ra_guess*10),
                                kwargs...) where {T<:Real}

    function growth_rate_at_Ra(Ra)
        params = OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=m, lmax=lmax, Nr=Nr; kwargs...)
        op = LinearStabilityOperator(params)
        σ, _, _ = find_growth_rate(op)
        return σ
    end

    Ra_low, Ra_high = Ra_bracket
    σ_low = growth_rate_at_Ra(Ra_low)
    σ_high = growth_rate_at_Ra(Ra_high)

    attempt = 0
    while σ_low * σ_high > 0 && attempt < 10
        if σ_low > 0
            Ra_low /= 2
            σ_low = growth_rate_at_Ra(Ra_low)
        else
            Ra_high *= 2
            σ_high = growth_rate_at_Ra(Ra_high)
        end
        attempt += 1
    end

    σ_low * σ_high > 0 && error("Could not bracket the critical Rayleigh number")

    prob = BracketingNonlinearProblem((Ra, _) -> growth_rate_at_Ra(Ra), (Ra_low, Ra_high), nothing)
    sol = solve(prob, Brent(); abstol=tol, reltol=tol, maxiters=200)
    Ra_c = T(sol.u)

    params_c = OnsetParams(E=E, Pr=Pr, Ra=Ra_c, χ=χ, m=m, lmax=lmax, Nr=Nr; kwargs...)
    op_c = LinearStabilityOperator(params_c)
    σ_c, ω_c, vec_c = find_growth_rate(op_c)

    return Ra_c, ω_c, vec_c
end

export OnsetParams, LinearStabilityOperator
export assemble_matrices
export solve_eigenvalue_problem, find_growth_rate, find_critical_rayleigh
