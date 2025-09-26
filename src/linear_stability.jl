module LinearStability

using LinearAlgebra
using SparseArrays
using SHTnsKit

import ..Cross: ChebyshevDiffn

export ShellParams, build_generalized_problem, leading_modes, critical_Rayleigh_search

struct ShellParams{T<:Real}
    m::Int
    E::T
    Pr::T
    Ra::T
    ri::T
    ro::T
    lmax::Int
    Nr::Int
end

function ShellParams(; m::Int, E::Real, Pr::Real, Ra::Real, ri::Real, ro::Real, lmax::Int, Nr::Int)
    return ShellParams{Float64}(m, float(E), float(Pr), float(Ra), float(ri), float(ro), lmax, Nr)
end

function _chebyshev_operators(params::ShellParams)
    cd = ChebyshevDiffn(params.Nr, [params.ri, params.ro], 2)
    return cd.x, cd.D₁, cd.D₂
end

function _angular_eigenvalues(params::ShellParams)
    cfg = create_gauss_config(params.lmax, params.lmax + 1; mmax=params.m, mres=1, nlon=max(2*params.m + 1, 4))
    Alm = zeros(ComplexF64, params.lmax + 1, params.m + 1)
    ones_vec = ones(ComplexF64, params.lmax + 1)
    @inbounds for ℓ in params.m:params.lmax
        Alm[ℓ+1, params.m+1] = 1.0 + 0im
    end
    dist_apply_laplacian!(cfg, Alm)
    λ = similar(ones_vec, params.lmax + 1)
    @inbounds for ℓ in params.m:params.lmax
        λ[ℓ+1] = -real(Alm[ℓ+1, params.m+1])
    end
    return λ
end

function build_generalized_problem(params::ShellParams)
    r, D1, D2 = _chebyshev_operators(params)
    inv_r2 = 1.0 ./ (r .^ 2)
    nr = length(r)

    λℓ = _angular_eigenvalues(params)
    ℓvals = collect(params.m:params.lmax)
    nℓ = length(ℓvals)
    blk = 3 * nr
    dim = blk * nℓ

    A = spzeros(ComplexF64, dim, dim)
    B = spzeros(ComplexF64, dim, dim)

    Gfac = params.ri * params.ro / (params.ro - params.ri)
    Ra_fac = params.Ra / params.Pr

    for (idx, ℓ) in enumerate(ℓvals)
        Lh = λℓ[ℓ+1]
        Lrad = D2 .- Diagonal(Lh .* inv_r2)
        C = Diagonal(2im * params.m .* inv_r2)
        block_start = (idx - 1) * blk + 1
        block_idx = block_start:(block_start+blk-1)

        Inr = Diagonal(fill(1.0 + 0im, nr))

        A11 = params.E .* (Lrad * Lrad) + Ra_fac .* Lh .* Inr
        A12 = C
        A13 = spzeros(ComplexF64, nr, nr)

        A21 = C
        A22 = Lrad
        A23 = spzeros(ComplexF64, nr, nr)

        G = Diagonal(-Gfac .* (Lh .* inv_r2 .* inv_r2))
        A31 = G
        A32 = spzeros(ComplexF64, nr, nr)
        A33 = (params.E / params.Pr) .* Lrad

        blockA = [A11  A12  A13;
                  A21  A22  A23;
                  A31  A32  A33]

        B11 = Lrad
        B22 = Lrad
        B33 = Inr
        blockB = [B11  spzeros(ComplexF64, nr, nr)  spzeros(ComplexF64, nr, nr);
                  spzeros(ComplexF64, nr, nr)  B22  spzeros(ComplexF64, nr, nr);
                  spzeros(ComplexF64, nr, nr)  spzeros(ComplexF64, nr, nr)  B33]

        _impose_velocity_bc!(blockA, blockB, D1, nr)
        _impose_toroidal_bc!(blockA, blockB, nr)
        _impose_temperature_bc!(blockA, blockB, nr)

        A[block_idx, block_idx] = sparse(blockA)
        B[block_idx, block_idx] = sparse(blockB)
    end

    return A, B, r, ℓvals
end

function _impose_velocity_bc!(A::AbstractMatrix, B::AbstractMatrix, D1::AbstractMatrix, nr::Int)
    # P(r=ri) = 0, ∂rP(r=ri)=0, ∂rP(r=ro)=0, P(r=ro)=0
    A[1, :] .= 0; A[1, 1] = 1
    B[1, :] .= 0
    A[2, :] .= D1[1, :]
    B[2, :] .= 0
    A[nr-1, :] .= D1[end, :]
    B[nr-1, :] .= 0
    A[nr, :] .= 0; A[nr, nr] = 1
    B[nr, :] .= 0
end

function _impose_toroidal_bc!(A::AbstractMatrix, B::AbstractMatrix, nr::Int)
    offset = nr
    rows = offset + (1:nr)
    A[rows[1], :] .= 0; A[rows[1], rows[1]] = 1; B[rows[1], :] .= 0
    A[rows[end], :] .= 0; A[rows[end], rows[end]] = 1; B[rows[end], :] .= 0
end

function _impose_temperature_bc!(A::AbstractMatrix, B::AbstractMatrix, nr::Int)
    offset = 2 * nr
    rows = offset + (1:nr)
    A[rows[1], :] .= 0; A[rows[1], rows[1]] = 1; B[rows[1], :] .= 0
    A[rows[end], :] .= 0; A[rows[end], rows[end]] = 1; B[rows[end], :] .= 0
end

function leading_modes(params::ShellParams; nev::Int=6, which::Symbol=:LR)
    A, B, _, ℓvals = build_generalized_problem(params)
    vals, vecs = eigs(A, B; nev=nev, which=which)
    return vals, vecs, ℓvals
end

function critical_Rayleigh_search(params::ShellParams; guess::Real, bracket::Tuple{Real,Real}, tol::Real=1e-6, maxiter::Int=25)
    left, right = bracket
    f(ra) = begin
        p = ShellParams(; m=params.m, E=params.E, Pr=params.Pr, Ra=ra, ri=params.ri, ro=params.ro, lmax=params.lmax, Nr=params.Nr)
        λ, _, _ = leading_modes(p; nev=1, which=:LR)
        return real(λ[1])
    end
    f_left = f(left)
    f_right = f(right)
    for k in 1:maxiter
        mid = (left + right) / 2
        f_mid = f(mid)
        if abs(f_mid) < tol
            return mid, f_mid
        end
        if sign(f_mid) == sign(f_left)
            left, f_left = mid, f_mid
        else
            right, f_right = mid, f_mid
        end
    end
    return (left + right)/2, f((left + right)/2)
end

end
