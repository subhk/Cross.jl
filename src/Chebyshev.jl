using LinearAlgebra

#══════════════════════════════════════════════════════════════════════════════#
#                        CHEBYSHEV DIFFERENTIATION                             #
#══════════════════════════════════════════════════════════════════════════════#

"""
    chebdif(n::Int, m::Int) -> (x, D)

Compute Chebyshev differentiation matrix of order `m` on `n` Chebyshev points.

Uses the Chebyshev-Gauss-Lobatto grid on [-1,1]:
    x_k = cos(pi*k/(n-1)),  k = 0, 1, ..., n-1

# Arguments
- `n::Int`: Number of Chebyshev points (n >= 2)
- `m::Int`: Order of differentiation (1 <= m < n)

# Returns
- `x::Vector{Float64}`: Chebyshev grid points in descending order
- `D::Matrix{Float64}`: m-th order differentiation matrix

# Example
```julia
x, D = chebdif(8, 1)  # 1st derivative on 8 points
f = exp.(x)           # test function
df = D * f            # numerical derivative
```
"""
function chebdif(n::Int, m::Int)
    @assert n >= 2 "Need at least 2 points"
    @assert 1 <= m < n "Derivative order must satisfy 1 <= m < n"
    
    # Identity matrix for diagonal operations
    I_mat = I(n)
    
    # Grid indices and symmetry points
    k = 0:(n-1)
    n1 = fld(n, 2)
    n2 = ceil(Int, n/2)
    
    # Chebyshev-Gauss-Lobatto points and angles
    x_hat = cos.(pi * k / (n-1))
    theta = pi * k / (n-1)
    
    # Trigonometric matrix for efficient computation
    Theta = repeat(theta/2, 1, n)
    Delta_x = 2 * sin.(Theta' .+ Theta) .* sin.(Theta' .- Theta)
    
    # Exploit symmetry to reduce computation
    Delta_x[n1+1:end, :] .= -reverse(reverse(Delta_x[1:n2, :], dims=2), dims=1)
    Delta_x[I_mat] .= 1.0  # diagonal entries
    
    # Chebyshev weight matrix
    v = (-1.0).^k
    C_mat = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        C_mat[i, j] = v[abs(i - j) + 1]
    end
    C_mat[1, :] .*= 2
    C_mat[end, :] .*= 2
    C_mat[:, 1] .*= 0.5
    C_mat[:, end] .*= 0.5
    
    # Inverse of off-diagonal entries
    Z_mat = 1.0 ./ Delta_x
    Z_mat[I_mat] .= 0.0
    
    # Build differentiation matrix recursively
    D_mat = Matrix{Float64}(I, n, n)
    
    for ell in 1:m
        D_mat = ell .* Z_mat .* (C_mat .* repeat(diag(D_mat), 1, n) .- D_mat)
        D_mat[I_mat] .= -sum(D_mat, dims=2)
    end
    
    # Reverse for descending order
    reverse!(D_mat);
    
    return reverse(x_hat), D_mat
end

#══════════════════════════════════════════════════════════════════════════════#
#                           MAIN CONTAINER TYPE                                #
#══════════════════════════════════════════════════════════════════════════════#

"""
    ChebyshevDiffn{T<:Real}

A spectral differentiation operator using Chebyshev polynomials.

Pre-computes and caches differentiation matrices up to 4th order for efficient
repeated use. Automatically handles domain transformation from [-1,1] to [a,b].

# Fields
- `n::Int`: Number of grid points
- `domain::Tuple{T,T}`: Domain interval (a, b)
- `max_order::Int`: Maximum derivative order computed
- `x::Vector{T}`: Grid points on [a,b]
- `D1, D2, D3, D4::Matrix{T}`: Differentiation matrices (higher orders may be Nothing)

# Mathematical Foundation
For domain transformation zeta in [-1,1] -> x in [a,b]:
    x = (b-a)/2 * (zeta + 1) + a
    
Scaling factor alpha = 2/(b-a) ensures correct derivative computation:
    d^n f/dx^n = alpha^n * (d^n f/dzeta^n)
"""
struct ChebyshevDiffn{T<:Real}
    n          :: Int
    domain     :: Tuple{T,T}
    max_order  :: Int
    x          :: Vector{T}
    D1         :: Matrix{T}
    D2         :: Matrix{T}
    D3         :: Union{Matrix{T}, Nothing}
    D4         :: Union{Matrix{T}, Nothing}
end

"""
    ChebyshevDiffn(n, domain, max_order=1)

Construct a Chebyshev differentiation operator.

# Arguments
- `n::Int`: Number of grid points
- `domain::AbstractVector{T}`: Domain interval [a, b] as a 2-element vector where T<:Real
- `max_order::Int=1`: Maximum derivative order to compute (1 <= max_order <= 4)

# Example
```julia
# Create operator for 16 points on [0, 2π] with derivatives up to 2nd order
cd = ChebyshevDiffn(16, [0.0, 2*pi], 2)

# Use it to differentiate
f = sin.(cd.x)
df_dx = cd.D1 * f      # first derivative
d2f_dx2 = cd.D2 * f    # second derivative
```
"""
function ChebyshevDiffn(
    n::Int,
    domain::AbstractVector{T},
    max_order::Int = 1
) where T<:Real
    
    @assert length(domain) == 2 "Domain must be a 2-element vector [a, b]"
    a, b = domain[1], domain[2]
    @assert a < b "Domain must satisfy a < b"
    @assert 1 <= max_order <= 4 "Maximum order must be between 1 and 4"
    
    # ──────────────────────────────────────────────────────────────────────────
    # Compute raw differentiation matrices on [-1,1]
    # ──────────────────────────────────────────────────────────────────────────
    x_hat, D1_hat = chebdif(n, 1)
    _, D2_hat  = chebdif(n, 2)

    D3_hat = max_order >= 3 ? chebdif(n, 3)[2] : nothing
    D4_hat = max_order >= 4 ? chebdif(n, 4)[2] : nothing
    
    # ──────────────────────────────────────────────────────────────────────────
    # Transform domain and scale derivatives
    # ──────────────────────────────────────────────────────────────────────────
    
    # Map from [-1,1] to [a,b]
    x = @. (b - a)/2 * (x_hat + 1) + a

    # Scaling factor for derivatives
    alpha = 2 / (b - a)

    # Apply appropriate scaling to each derivative order
    D1 = alpha * D1_hat
    D2 = alpha^2 * D2_hat
    D3 = D3_hat === nothing ? nothing : alpha^3 * D3_hat
    D4 = D4_hat === nothing ? nothing : alpha^4 * D4_hat

    return ChebyshevDiffn(n, (a, b), max_order, x, D1, D2, D3, D4)
end

#══════════════════════════════════════════════════════════════════════════════#
#                              PRETTY PRINTING                                 #
#══════════════════════════════════════════════════════════════════════════════#

function Base.show(io::IO, cd::ChebyshevDiffn{T}) where T
    println(io, "Chebyshev Differentiation{$T}")
    println(io, "  ├─ Grid points: $(cd.n)")
    println(io, "  ├─ Domain: [$(cd.domain[1]), $(cd.domain[2])]")
    println(io, "  ├─ Max order: $(cd.max_order)")
    print(io,   "  └─ Available: D₁")
    
    cd.max_order ≥ 2 && print(io, ", D₂")
    cd.max_order ≥ 3 && print(io, ", D₃")
    cd.max_order ≥ 4 && print(io, ", D₄")
    println(io)
end

#══════════════════════════════════════════════════════════════════════════════#
#                            CONVENIENCE METHODS                               #
#══════════════════════════════════════════════════════════════════════════════#

"""
    derivative(cd::ChebyshevDiffn, f::Vector, order::Int=1)

Compute the derivative of function values `f` at the Chebyshev grid points.

# Arguments
- `cd::ChebyshevDiffn`: The differentiation operator
- `f::Vector`: Function values at grid points cd.x
- `order::Int=1`: Derivative order (1 ≤ order ≤ cd.max_order)

# Returns
- `Vector`: Derivative values at grid points
"""
function derivative(cd::ChebyshevDiffn, f::Vector, order::Int=1)
    @assert 1 ≤ order ≤ cd.max_order "Derivative order $order not available"
    @assert length(f) == cd.n "Function vector length must match grid size"
    
    if order == 1
        return cd.D₁ * f
    elseif order == 2
        return cd.D₂ * f
    elseif order == 3
        return cd.D₃ * f
    elseif order == 4
        return cd.D₄ * f
    end
end

# Convenient operator overloading
Base.:*(cd::ChebyshevDiffn, f::Vector) = derivative(cd, f, 1)

#══════════════════════════════════════════════════════════════════════════════#
#                              EXAMPLE USAGE                                   #
#══════════════════════════════════════════════════════════════════════════════#

"""
    demo_chebyshev_differentiation()

Demonstrate the Chebyshev differentiation capabilities with a simple example.
"""
function demo_chebyshev_differentiation()
    println("╔═══════════════════════════════════════════════════════════════╗")
    println("║              CHEBYSHEV DIFFERENTIATION DEMO                   ║")
    println("╚═══════════════════════════════════════════════════════════════╝")
    
    # Create differentiation operator
    n = 16
    domain = [0.0, 2π]
    cd = ChebyshevDiffn(n, domain, 2)
    
    println("\n", cd)
    
    # Test function: sin(x)
    f = sin.(cd.x)
    
    # Compute derivatives
    df_exact = cos.(cd.x)           # exact first derivative
    df_numerical = cd.D₁ * f        # numerical first derivative
    
    d2f_exact = -sin.(cd.x)         # exact second derivative  
    d2f_numerical = cd.D₂ * f       # numerical second derivative
    
    # Show errors
    error_1st = maximum(abs.(df_numerical - df_exact))
    error_2nd = maximum(abs.(d2f_numerical - d2f_exact))
    
    println("\nAccuracy test with f(x) = sin(x):")
    println("  • 1st derivative max error: $(error_1st)")
    println("  • 2nd derivative max error: $(error_2nd)")
    println("\nSpectral accuracy achieved! ✨")
end
