using LinearAlgebra

# =============================================================================
#  Chebyshev Differentiation Matrices 
#
#  This module constructs spectral differentiation matrices
#  The approach:
#    1. Work with Chebyshev–Gauss–Lobatto nodes on [-1, 1].
#    2. Build the Vandermonde matrix V relating Chebyshev coefficients to
#       values at those nodes.
#    3. Form derivative operators in coefficient space via the recursive
#       Dcheb algorithm 
#    4. Convert coefficient derivatives back to value space:
#           Dₖ = V * Dₖ^(coeff) * V⁻¹
#    5. Apply affine scaling for arbitrary physical domains [a, b].
# =============================================================================

"""
    ChebyshevDiffn{T}

Container holding Chebyshev nodes and spectral differentiation matrices up to
fourth order on an arbitrary interval `[a, b]`.

Fields
------
- `n`           : number of collocation points
- `domain`      : `(a, b)` interval
- `max_order`   : highest derivative order constructed
- `x`           : physical nodes on `[a, b]`
- `D1`          : first-derivative matrix acting on nodal values
- `D2`          : second-derivative matrix
- `D3`, `D4`    : third and fourth derivative matrices (or `nothing`)

All derivative matrices act on column vectors of nodal values arranged in
ascending order (from `x = a` to `x = b`).
"""
struct ChebyshevDiffn{T<:AbstractFloat}
    n::Int
    domain::Tuple{T,T}
    max_order::Int
    x::Vector{T}
    D1::Matrix{T}
    D2::Matrix{T}
    D3::Union{Matrix{T},Nothing}
    D4::Union{Matrix{T},Nothing}
end

# -----------------------------------------------------------------------------
#  Helper functions (internal)
# -----------------------------------------------------------------------------

"""Compute Chebyshev-Gauss-Lobatto nodes on `[-1, 1]` in ascending order."""
chebyshev_nodes(n::Int) = cospi.(reverse(0:n-1) ./ (n-1))

"""Build the Vandermonde matrix for Chebyshev polynomials at normalized nodes."""
function chebyshev_vandermonde(x̂::AbstractVector{T}) where {T<:Real}
    n = length(x̂)
    V = Matrix{T}(undef, n, n)
    θ = acos.(x̂)
    @inbounds for j in 1:n
        V[:, j] = cos.((j - 1) .* θ)
    end
    return V
end

"""
    chebyshev_coeff_derivative!(out, coeff)

Overwrite `out` with the Chebyshev coefficients of the derivative of `coeff`.
"""
function chebyshev_coeff_derivative!(out::Vector{T}, coeff::Vector{T}) where {T<:AbstractFloat}
    s = length(coeff)
    fill!(out, zero(T))
    s <= 1 && return out

    out[s - 1] = T(2) * (s - 1) * coeff[s]
    for k in (s - 2):-1:1
        out[k] = out[k + 2] + T(2) * k * coeff[k + 1]
    end
    out[1] /= T(2)
    return out
end

"""Construct the coefficient-space Chebyshev derivative matrix on `[-1, 1]`."""
function chebyshev_coeff_derivative_matrix(n::Int)
    D = zeros(Float64, n, n)
    scratch = zeros(Float64, n)
    coeff = zeros(Float64, n)
    for j in 1:n
        fill!(coeff, 0.0)
        coeff[j] = 1.0
        chebyshev_coeff_derivative!(scratch, coeff)
        D[:, j] .= scratch
    end
    return D
end

"""Convert precomputed Float64 derivative matrices to the requested real type."""
promote_matrix(::Type{T}, M::Matrix{Float64}) where {T<:Real} = Matrix{T}(M)

# -----------------------------------------------------------------------------
#  Constructor
# -----------------------------------------------------------------------------

"""Construct Chebyshev nodes and differentiation matrices on a physical interval."""
function ChebyshevDiffn(n::Int, domain::AbstractVector{T}, max_order::Int = 1) where {T<:AbstractFloat}
    length(domain) == 2 || throw(ArgumentError(
        "Domain must be specified as [a, b], got length $(length(domain))"))
    n ≥ 2 || throw(ArgumentError(
        "Need at least two Chebyshev points, got $n"))
    1 ≤ max_order ≤ 4 || throw(ArgumentError(
        "Supported derivative orders: 1 ≤ max_order ≤ 4, got $max_order"))

    a, b = domain[1], domain[2]
    a < b || throw(ArgumentError(
        "Domain must satisfy a < b, got a=$a, b=$b"))

    # Nodes on [-1,1] and corresponding Vandermonde matrix
    x̂ = chebyshev_nodes(n)
    V = chebyshev_vandermonde(x̂)

    # Use LU factorization for more stable computation of V \ (...)
    V_lu = lu(V)

    # Coefficient-space derivative matrices
    Dc1 = chebyshev_coeff_derivative_matrix(n)
    Dc2 = Dc1 * Dc1
    Dc3 = Dc1 * Dc2
    Dc4 = Dc1 * Dc3

    # Convert to value-space operators (on [-1, 1]) using D = V * Dc * V^{-1}
    # Computed as (V * Dc) / V via LU factorization for stability
    D1_hat = (V * Dc1) / V_lu
    D2_hat = (V * Dc2) / V_lu
    D3_hat = (V * Dc3) / V_lu
    D4_hat = (V * Dc4) / V_lu

    # Map nodes to [a, b] and scale derivatives
    x = @. (b - a) / 2 * (x̂ + 1) + a
    α = 2 / (b - a)

    D1 = α * D1_hat
    D2 = α^2 * D2_hat
    D3 = max_order ≥ 3 ? α^3 * D3_hat : nothing
    D4 = max_order ≥ 4 ? α^4 * D4_hat : nothing

    return ChebyshevDiffn(
        n,
        (T(a), T(b)),
        max_order,
        Vector{T}(x),
        promote_matrix(T, D1),
        promote_matrix(T, D2),
        D3 === nothing ? nothing : promote_matrix(T, D3),
        D4 === nothing ? nothing : promote_matrix(T, D4)
    )
end

"""Print a compact summary of a Chebyshev differentiation cache."""
function Base.show(io::IO, cd::ChebyshevDiffn{T}) where {T}
    println(io, "ChebyshevDiffn{$T}")
    println(io, "  points    : ", cd.n)
    println(io, "  domain    : [", cd.domain[1], ", ", cd.domain[2], "]")
    println(io, "  max order : ", cd.max_order)
    available = ["D1", "D2"]
    cd.max_order ≥ 3 && push!(available, "D3")
    cd.max_order ≥ 4 && push!(available, "D4")
    println(io, "  matrices  : ", join(available, ", "))
end
