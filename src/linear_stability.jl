module LinearStability

using LinearAlgebra
using SparseArrays
using SHTnsKit
using Arpack: eigs

import ..Cross: ChebyshevDiffn

export ShellParams, build_generalized_problem, leading_modes, critical_Rayleigh_search

"""
    ShellParams(; m, E, Pr, Ra, ri, ro, lmax, Nr)

Container for the dimensionless control parameters entering the linear
stability problem described by Equations (10)–(19) of the Onset of
Convection reference in `docs/Onset_convection.pdf`. The fields correspond to:

  • `m`   – fixed azimuthal wavenumber of the perturbation (Equation 12)
  • `E`   – Ekman number multiplying viscous terms in Equation (10)
  • `Pr`  – Prandtl number coupling momentum and heat Equations (10)–(11)
  • `Ra`  – Rayleigh number prefactor of the buoyancy term in Equation (10)
  • `ri`, `ro` – nondimensional radii of the spherical shell boundaries
  • `lmax` – spherical harmonic truncation in the expansion (Equations 13–15)
  • `Nr`  – number of Chebyshev collocation points across the shell

The constructor normalises all floating-point inputs to `Float64` so the
downstream sparse assembly works with a consistent eltype.
"""
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
    @inbounds for ℓ in params.m:params.lmax
        Alm[ℓ+1, params.m+1] = 1.0 + 0im
    end
    dist_apply_laplacian!(cfg, Alm)
    λ = zeros(Float64, params.lmax + 1)
    @inbounds for ℓ in params.m:params.lmax
        λ[ℓ+1] = -real(Alm[ℓ+1, params.m+1])
    end
    return λ
end

"""
    build_generalized_problem(params) -> (A, B, r, ℓvals)

Assemble the sparse matrices `A` and `B` of the generalized eigenvalue
problem `A x = λ B x` obtained by applying the poloidal–toroidal
decomposition (Equation 13) to the linearised dynamics (Equations 10–11).

Implementation details:
  • Radial derivatives use Chebyshev collocation on `[ri, ro]` via
    `ChebyshevDiffn` (Equation 15).
  • Angular couplings are diagonal because the SHTnsKit helpers provide
    the eigenvalues `ℓ(ℓ+1)` of the spherical Laplacian acting on
    the Schmidt semi-normalised harmonics (Equations 14 & 19).
  • No-slip, fixed-temperature boundary conditions (Equations 16–18) are
    enforced by replacing boundary rows with tau constraints.

Returns the assembled matrices together with the radial grid `r` and the
list of active spherical degrees `ℓvals` for convenience.
"""
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

function _apply_dirichlet!(A::AbstractMatrix, B::AbstractMatrix, idx::Int)
    A[idx, :] .= 0
    A[:, idx] .= 0
    A[idx, idx] = 1
    B[idx, :] .= 0
    B[:, idx] .= 0
end

function _impose_velocity_bc!(A::AbstractMatrix, B::AbstractMatrix, D1::AbstractMatrix, nr::Int)
    # P(r=ri) = 0, ∂rP(r=ri)=0, ∂rP(r=ro)=0, P(r=ro)=0
    _apply_dirichlet!(A, B, 1)
    A[2, :] .= D1[1, :]
    B[2, :] .= 0
    A[nr-1, :] .= D1[end, :]
    B[nr-1, :] .= 0
    _apply_dirichlet!(A, B, nr)
end

function _impose_toroidal_bc!(A::AbstractMatrix, B::AbstractMatrix, nr::Int)
    offset = nr
    rows = offset + (1:nr)
    _apply_dirichlet!(A, B, rows[1])
    _apply_dirichlet!(A, B, rows[end])
end

function _impose_temperature_bc!(A::AbstractMatrix, B::AbstractMatrix, nr::Int)
    offset = 2 * nr
    rows = offset + (1:nr)
    _apply_dirichlet!(A, B, rows[1])
    _apply_dirichlet!(A, B, rows[end])
end

"""
    leading_modes(params; nev=6, which=:LR)

Solve the assembled eigenvalue problem for the `nev` eigenvalues of largest
real part (default) using ARPACK.  The eigenvalues correspond to the growth
rates `λ = σ + iω` defined in Equation (12); neutral stability occurs when
`σ = 0`.  Returns `(λ, v, ℓvals)` where `ℓvals` reuses the indices from
`build_generalized_problem`.
"""
function leading_modes(params::ShellParams; nev::Int=6, which::Symbol=:LR)
    A, B, _, ℓvals = build_generalized_problem(params)
    vals, vecs = eigs(A, B; nev=nev, which=which)
    return vals, vecs, ℓvals
end

"""
    critical_Rayleigh_search(params; bracket, tol=1e-6, maxiter=25)

Locate the Rayleigh number at which the leading growth rate changes sign
for the supplied azimuthal symmetry and Ekman/Prandtl numbers.  The
function performs a safeguarded bisection on the real part of the dominant
eigenvalue returned by `leading_modes`, following the procedure outlined
in Section 3.1 of the reference.

Arguments:
  • `bracket = (Ra_lo, Ra_hi)` must straddle the critical Rayleigh number.
  • `tol` and `maxiter` stop the bisection once `|σ| < tol` or the
    iterations are exhausted.

Returns the estimated `Rac` together with the final growth rate value.
"""
function critical_Rayleigh_search(params::ShellParams; bracket::Tuple{Real,Real}, tol::Real=1e-6, maxiter::Int=25)
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
