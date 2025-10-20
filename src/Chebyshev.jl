using LinearAlgebra

# =============================================================================
#  Chebyshev Differentiation Matrices (Kore-compatible)
#
#  This module constructs spectral differentiation matrices whose discrete
#  derivatives exactly match the coefficient-based formulation used in Kore.
#  The approach:
#    1. Work with Chebyshev–Gauss–Lobatto nodes on [-1, 1].
#    2. Build the Vandermonde matrix V relating Chebyshev coefficients to
#       values at those nodes.
#    3. Form derivative operators in coefficient space via the recursive
#       Dcheb algorithm (ported from kore-main/bin/utils.py).
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

All derivative matrices act on column vectors of nodal values arranged in the
standard Chebyshev–Gauss–Lobatto ordering (descending `x`).
"""
struct ChebyshevDiffn{T<:Real}
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

# Compute Chebyshev–Gauss–Lobatto nodes on [-1, 1] in descending order.
chebyshev_nodes(n::Int) = cospi.(reverse(0:n-1) ./ (n-1))

# Vandermonde matrix evaluating Chebyshev polynomials T_k(x) at nodes x̂.
function chebyshev_vandermonde(x̂::Vector{Float64})
    n = length(x̂)
    V = Array{Float64}(undef, n, n)
    θ = acos.(x̂)
    @inbounds for j in 1:n
        V[:, j] = cos.((j - 1) .* θ)
    end
    return V
end

# Derivative of Chebyshev coefficients (ported from Kore's utils.Dcheb).
function chebyshev_coeff_derivative!(out::Vector{Float64}, coeff::Vector{Float64})
    s = length(coeff)
    fill!(out, 0.0)
    s <= 1 && return out

    tmp0 = copy(coeff)
    tmp0[1] *= 2.0

    out[s - 1] = 2.0 * (s - 1) * tmp0[s]
    for k in (s - 2):-1:1
        out[k] = out[k + 2] + 2.0 * k * coeff[k + 1]
    end
    out[1] /= 2.0
    return out
end

# Matrix mapping Chebyshev coefficients to derivatives (on [-1, 1]).
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

# Promote Float64 matrices to element type T.
promote_matrix(::Type{T}, M::Matrix{Float64}) where {T<:Real} = Matrix{T}(M)

# -----------------------------------------------------------------------------
#  Constructor
# -----------------------------------------------------------------------------

function ChebyshevDiffn(n::Int, domain::AbstractVector{T}, max_order::Int = 1) where {T<:Real}
    @assert length(domain) == 2 "Domain must be specified as [a, b]"
    @assert n ≥ 2 "Need at least two Chebyshev points"
    @assert 1 ≤ max_order ≤ 4 "Supported derivative orders: 1 ≤ max_order ≤ 4"

    a, b = domain[1], domain[2]
    @assert a < b "Domain must satisfy a < b"

    # Nodes on [-1,1] and corresponding Vandermonde matrices
    x̂ = chebyshev_nodes(n)
    V = chebyshev_vandermonde(x̂)
    V_inv = inv(V)

    # Coefficient-space derivative matrices
    Dc1 = chebyshev_coeff_derivative_matrix(n)
    Dc2 = Dc1 * Dc1
    Dc3 = Dc1 * Dc2
    Dc4 = Dc1 * Dc3

    # Convert to value-space operators (on [-1, 1])
    D1_hat = V * Dc1 * V_inv
    D2_hat = V * Dc2 * V_inv
    D3_hat = V * Dc3 * V_inv
    D4_hat = V * Dc4 * V_inv

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
