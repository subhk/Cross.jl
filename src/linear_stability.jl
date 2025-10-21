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
    use_kore_weighting::Bool = true
    equatorial_symmetry::Symbol = :both
    basic_state::Nothing = nothing

    function OnsetParams{T}(E, Pr, Ra, χ, m, lmax, Nr, ri, ro, L,
                           mechanical_bc, thermal_bc, use_kore_weighting,
                           equatorial_symmetry,
                           basic_state) where {T}
        @assert 0 < χ < 1 "Radius ratio must satisfy 0 < χ < 1"
        @assert E > 0 "Ekman number must be positive"
        @assert Pr > 0 "Prandtl number must be positive"
        @assert m ≥ 0 "Azimuthal wavenumber must be non-negative"
        @assert lmax ≥ m "lmax must be ≥ m"
        @assert Nr ≥ 4 "Need at least 4 radial points"
        @assert mechanical_bc in (:no_slip, :stress_free) "Invalid mechanical BC"
        @assert thermal_bc in (:fixed_temperature, :fixed_flux) "Invalid thermal BC"
        @assert equatorial_symmetry in (:both, :symmetric, :antisymmetric) "equatorial_symmetry must be :both, :symmetric, or :antisymmetric"

        new{T}(E, Pr, Ra, χ, m, lmax, Nr, ri, ro, L,
               mechanical_bc, thermal_bc, use_kore_weighting, equatorial_symmetry,
               basic_state)
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

    TT = eltype(op.r)

    for ℓ in p.m:p.lmax
        L = TT(ℓ * (ℓ + 1))

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
            Cminus = (ℓ^2 - 1) * sqrt(max(zero(TT), ℓ^2 - m^2)) / (2ℓ - 1)
            coupling = 2 * Cminus * ((ℓ - 1) * R3D0 - R4D1)
            A[P_idx, op.index_map[(ℓ-1, :T)]] .+= Complex.(coupling)
        end
        if ℓ < p.lmax
            Cplus = ℓ*(ℓ+2) * sqrt(max(zero(TT), (ℓ+m+1)*(ℓ-m+1))) / (2ℓ + 3)
            coupling = 2 * Cplus * (-(ℓ + 2) * R3D0 - R4D1)
            A[P_idx, op.index_map[(ℓ+1, :T)]] .+= Complex.(coupling)
        end

        # Toroidal diagonal
        coriolis_t = -2im * m * L * R2D0
        viscous_t = Ek * L * (-L * R0 + 2 * R1D1 + R2D2)
        A[T_idx, T_idx] .+= Complex.(coriolis_t - viscous_t)

        # Toroidal ↔ poloidal coupling
        if ℓ > p.m
            Cminus = (ℓ^2 - 1) * sqrt(max(zero(TT), ℓ^2 - m^2)) / (2ℓ - 1)
            coupling = 2 * Cminus * ((ℓ - 1) * R1D0 - R2D1)
            A[T_idx, op.index_map[(ℓ-1, :P)]] .+= Complex.(coupling)
        end
        if ℓ < p.lmax
            Cplus = ℓ*(ℓ+2) * sqrt(max(zero(TT), (ℓ+m+1)*(ℓ-m+1))) / (2ℓ + 3)
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
                                  nev::Int=1,
                                  tol::Float64=1e-10,
                                  maxiter::Int=1000,
                                  which::Symbol=:LR) where {T<:Float64}
    A, B, _, _ = assemble_matrices(op)
    σ = 0.1 + 0.1im

    lu_factor = lu(A - σ * B)
    tmp = zeros(Complex{T}, size(A, 1))
    shift_map = LinearMap{Complex{T}}(ShiftInvertMap(lu_factor, B, tmp), 
                                size(A, 1); ismutating=true)

    krylovdim = min(size(A, 1), max(nev * 8, 60))
    x0 = randn(Complex{T}, size(A, 1))

    vals_inv, vecs, info = eigsolve(shift_map, 
                                    x0, nev, :LM;
                                    tol=tol, 
                                    maxiter=maxiter,
                                    krylovdim=500, 
                                    verbosity=0)

    finite = [abs(λ) > eps(T) for λ in vals_inv]
    any(finite) || error("No finite eigenvalues returned by eigensolver")
    vals_inv = vals_inv[finite]
    vecs = vecs[finite]

    eigenvalues = Complex{T}[σ + inv(λ) for λ in vals_inv]

    ordering = which == :LR ? sortperm(real.(eigenvalues); rev=true) :
               which == :LM ? sortperm(abs.(eigenvalues); rev=true) :
               collect(1:length(eigenvalues))

    return eigenvalues[ordering], vecs[ordering], info
end


function find_growth_rate(op::LinearStabilityOperator; kwargs...)
    eigenvalues, eigenvectors, info = solve_eigenvalue_problem(op; kwargs...)

    println("Eigenvalue solver converged: ", info.converged)
    println("Eigenvalues found: ", eigenvalues)

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
    m_int = Int(m)
    lmax ≥ m_int || error("lmax must be ≥ m (got lmax=$lmax, m=$m_int)")

    function growth_rate_at_Ra(Ra)
        params = OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=m_int, lmax=lmax, Nr=Nr; kwargs...)
        op = LinearStabilityOperator(params)
        σ, _, _ = find_growth_rate(op)
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
        params_c = OnsetParams(E=E, Pr=Pr, Ra=Ra_root, χ=χ, m=m_int, lmax=lmax, Nr=Nr; kwargs...)
        op_c = LinearStabilityOperator(params_c)
        σ_c, ω_c, vec_c = find_growth_rate(op_c)
        return Ra_root, ω_c, vec_c
    end

    Ra_low = convert(T, Ra_bracket[1])
    Ra_high = convert(T, Ra_bracket[2])
    state, Ra_root = add_sample!(Ra_low)
    if state == :root
        params_c = OnsetParams(E=E, Pr=Pr, Ra=Ra_root, χ=χ, m=m_int, lmax=lmax, Nr=Nr; kwargs...)
        op_c = LinearStabilityOperator(params_c)
        σ_c, ω_c, vec_c = find_growth_rate(op_c)
        return Ra_root, ω_c, vec_c
    end
    state, Ra_root = add_sample!(Ra_high)
    if state == :root
        params_c = OnsetParams(E=E, Pr=Pr, Ra=Ra_root, χ=χ, m=m_int, lmax=lmax, Nr=Nr; kwargs...)
        op_c = LinearStabilityOperator(params_c)
        σ_c, ω_c, vec_c = find_growth_rate(op_c)
        return Ra_root, ω_c, vec_c
    end

    attempt = 0
    while (pos[] === nothing || neg[] === nothing) && attempt < 12
        if pos[] === nothing
            Ra_high *= T(2)
            state, Ra_root = add_sample!(Ra_high)
            if state == :root
                params_c = OnsetParams(E=E, Pr=Pr, Ra=Ra_root, χ=χ, m=m_int, lmax=lmax, Nr=Nr; kwargs...)
                op_c = LinearStabilityOperator(params_c)
                σ_c, ω_c, vec_c = find_growth_rate(op_c)
                return Ra_root, ω_c, vec_c
            end
        end
        if neg[] === nothing
            Ra_low /= T(2)
            state, Ra_root = add_sample!(Ra_low)
            if state == :root
                params_c = OnsetParams(E=E, Pr=Pr, Ra=Ra_root, χ=χ, m=m_int, lmax=lmax, Nr=Nr; kwargs...)
                op_c = LinearStabilityOperator(params_c)
                σ_c, ω_c, vec_c = find_growth_rate(op_c)
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
                params_c = OnsetParams(E=E, Pr=Pr, Ra=Ra_root, χ=χ, m=m_int, lmax=lmax, Nr=Nr; kwargs...)
                op_c = LinearStabilityOperator(params_c)
                σ_c, ω_c, vec_c = find_growth_rate(op_c)
                return Ra_root, ω_c, vec_c
            end
            if pos[] !== nothing && neg[] !== nothing
                break
            end
            Ra_down = convert(T, 10.0^(log_guess - k * step))
            state, Ra_root = add_sample!(Ra_down)
            if state == :root
                params_c = OnsetParams(E=E, Pr=Pr, Ra=Ra_root, χ=χ, m=m_int, lmax=lmax, Nr=Nr; kwargs...)
                op_c = LinearStabilityOperator(params_c)
                σ_c, ω_c, vec_c = find_growth_rate(op_c)
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
    params_c = OnsetParams(E=E, Pr=Pr, Ra=Ra_c, χ=χ, m=m_int, lmax=lmax, Nr=Nr; kwargs...)
    op_c = LinearStabilityOperator(params_c)
    σ_c, ω_c, vec_c = find_growth_rate(op_c)

    return Ra_c, ω_c, vec_c
end

export OnsetParams, LinearStabilityOperator
export assemble_matrices
export solve_eigenvalue_problem, find_growth_rate, find_critical_rayleigh
