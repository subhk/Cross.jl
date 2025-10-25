# =============================================================================
#  Ultraspherical (Gegenbauer) Spectral Method
#
#  Implementation of the Olver-Townsend (2013) sparse spectral method
#  following the approach used in Kore for rotating spherical shell problems.
#
#  References:
#  - Olver & Townsend (2013), SIAM Review 55(3), 462-489
#  - Rekier et al. (2019), Kore implementation
# =============================================================================

module UltrasphericalSpectral

using LinearAlgebra
using SparseArrays

export ultraspherical_derivative,
       ultraspherical_conversion,
       sparse_radial_operator,
       chebyshev_grid,
       chebyshev_transform,
       apply_boundary_conditions!

# -----------------------------------------------------------------------------
# Chebyshev-Gauss-Lobatto grid and transforms
# -----------------------------------------------------------------------------

"""
    chebyshev_grid(N::Int) -> Vector{Float64}

Generate Chebyshev-Gauss-Lobatto grid points on [-1, 1].
Points are ordered from x = 1 (outer boundary) to x = -1 (inner boundary).
"""
function chebyshev_grid(N::Int)
    return [cos(π * j / N) for j in 0:N]
end

"""
    chebyshev_transform(f::Vector{T}) -> Vector{T}

Forward Chebyshev transform using DCT-I.
Converts function values at Chebyshev grid points to Chebyshev coefficients.
"""
function chebyshev_transform(f::Vector{T}) where {T<:Real}
    N = length(f) - 1
    # Use discrete cosine transform type I
    # This could be optimized with FFTW.jl
    coeffs = zeros(T, N+1)
    for n in 0:N
        s = zero(T)
        for j in 0:N
            xj = cos(π * j / N)
            Tn = cos(n * acos(xj))
            w = (j == 0 || j == N) ? 0.5 : 1.0
            s += w * f[j+1] * Tn
        end
        cn = (n == 0 || n == N) ? 2.0 : 1.0
        coeffs[n+1] = (2.0 / (cn * N)) * s
    end
    return coeffs
end

# -----------------------------------------------------------------------------
# Ultraspherical (Gegenbauer) polynomials
# -----------------------------------------------------------------------------

"""
    ultraspherical_conversion(λ::Real, N::Int) -> SparseMatrixCSC

Sparse conversion matrix S^(λ) that converts Chebyshev T_n coefficients
to ultraspherical C_n^(λ) coefficients.

If u = Σ a_n T_n(x), then u = Σ b_n C_n^(λ)(x) where b = S^(λ) * a.

For λ = 1/2, C_n^(1/2) are the Chebyshev polynomials of the second kind U_n.
For λ = 1, C_n^(1) are related to Legendre polynomials.
"""
function ultraspherical_conversion(λ::Real, N::Int)
    # Build sparse conversion matrix
    # This uses the recurrence relations between Chebyshev and ultraspherical

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    if λ == 0.0
        # Identity: C_n^(0) = T_n / (2n) for n > 0
        for n in 0:N
            push!(rows, n+1)
            push!(cols, n+1)
            push!(vals, n == 0 ? 1.0 : 1.0 / (2.0 * n))
        end
    else
        # General ultraspherical conversion
        # Based on: T_n = (1/2^λ) * Σ_{k=0}^{floor(n/2)} ...
        # For efficiency, we use the sparse structure

        for n in 0:N
            if n == 0
                push!(rows, 1)
                push!(cols, 1)
                push!(vals, 1.0)
            elseif n == 1
                push!(rows, 2)
                push!(cols, 2)
                push!(vals, 1.0 / (2.0 * λ))
            else
                # Diagonal term
                push!(rows, n+1)
                push!(cols, n+1)
                push!(vals, 1.0 / (2.0 * λ))

                # Off-diagonal term C_{n-2}^(λ)
                if n >= 2
                    push!(rows, n+1)
                    push!(cols, n-1)
                    push!(vals, -1.0 / (2.0 * λ))
                end
            end
        end
    end

    return sparse(rows, cols, vals, N+1, N+1)
end

"""
    ultraspherical_derivative(λ::Real, N::Int) -> SparseMatrixCSC

Sparse differentiation matrix in the ultraspherical basis.

If u = Σ a_n C_n^(λ)(x), then du/dx = Σ b_n C_n^(λ+1)(x) where b = D^(λ) * a.

The matrix D^(λ) is a sparse banded matrix with structure:
    b_n = 2(n+λ) a_{n+1}
"""
function ultraspherical_derivative(λ::Real, N::Int)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for n in 0:(N-1)
        # d/dx C_n^(λ) = 2λ C_{n-1}^(λ+1)  for n >= 1
        # But in coefficient space: if u = Σ a_k C_k^(λ)
        # then du/dx = Σ b_k C_k^(λ+1) where b_k = 2(k+λ) a_{k+1}

        push!(rows, n+1)
        push!(cols, n+2)  # b_n comes from a_{n+1}
        push!(vals, 2.0 * (n + λ))
    end

    return sparse(rows, cols, vals, N+1, N+1)
end

# -----------------------------------------------------------------------------
# Sparse radial operators
# -----------------------------------------------------------------------------

"""
    sparse_radial_operator(power::Int, deriv_order::Int, N::Int,
                           ri::Real, ro::Real) -> SparseMatrixCSC

Create sparse operator for r^power * D^deriv_order on [ri, ro].

This implements the Kore notation: r^k * d^n/dr^n

The method:
1. Map [ri, ro] → [-1, 1]
2. Apply derivative in ultraspherical basis (sparse)
3. Apply multiplication by powers of r (diagonal/sparse)
4. Convert back through ultraspherical chain
"""
function sparse_radial_operator(power::Int, deriv_order::Int, N::Int,
                                ri::Real, ro::Real)
    # Change of variables: r = ((ro-ri)*x + (ro+ri))/2, x ∈ [-1,1]
    # dr/dx = (ro-ri)/2
    L = ro - ri
    scale = 2.0 / L

    # Start with identity in Chebyshev basis
    D = sparse(1.0I, N+1, N+1)

    # Apply derivatives using ultraspherical chain
    λ = 0.0
    for k in 1:deriv_order
        # Convert to ultraspherical basis C^(λ)
        S = ultraspherical_conversion(λ, N)

        # Differentiate in ultraspherical basis
        Dλ = ultraspherical_derivative(λ, N)

        # Chain: d/dr = (dx/dr) d/dx = scale * d/dx
        D = (scale * Dλ) * S * D

        # Next derivative is in C^(λ+1) basis
        λ += 1.0
    end

    # If λ > 0, convert back to Chebyshev basis
    if λ > 0
        # Need inverse conversion
        S_inv = ultraspherical_conversion(λ, N)
        # Use sparse solve to keep sparsity
        D = sparse(S_inv \ Matrix(D))
    end

    # Apply r^power multiplication
    if power != 0
        # r(x) = ((ro-ri)*x + (ro+ri))/2
        # Multiplication by r^k in physical space corresponds to
        # a multiplication operator in spectral space
        # For now, use a simplified diagonal operator
        # TODO: Implement proper spectral multiplication using Clenshaw

        # Convert to dense for r-multiplication, then back to sparse
        x = chebyshev_grid(N)
        r_vals = @. ((ro - ri) * x + (ro + ri)) / 2.0
        R = spdiagm(0 => r_vals .^ power)

        # Keep sparse structure
        D_dense = Matrix(D)
        D = sparse(R * D_dense)
    end

    return D
end

"""
    apply_boundary_conditions!(A::SparseMatrixCSC, B::SparseMatrixCSC,
                               bc_rows::Vector{Int}, bc_type::Symbol)

Apply boundary conditions by replacing rows in the matrices A and B.

bc_type can be:
  - :dirichlet → u = 0
  - :neumann → du/dr = 0
  - :no_slip → u = du/dr = 0 (for poloidal)
  - :stress_free → u = d²u/dr² - 2u/r² = 0
"""
function apply_boundary_conditions!(A::SparseMatrixCSC, B::SparseMatrixCSC,
                                   bc_rows::Vector{Int}, bc_type::Symbol,
                                   N::Int, ri::Real, ro::Real)
    # Tau method: replace rows corresponding to boundary conditions

    for row in bc_rows
        # Zero out the row
        A[row, :] .= 0.0
        B[row, :] .= 0.0

        if bc_type == :dirichlet
            # u(r_boundary) = 0
            # Set row to evaluation at boundary point
            if row == 1  # Outer boundary (x = 1, r = ro)
                for n in 0:N
                    Tn = cos(n * 0.0)  # T_n(1) = 1 for all n
                    A[row, n+1] = Tn
                end
            elseif row == N+1  # Inner boundary (x = -1, r = ri)
                for n in 0:N
                    Tn = cos(n * π)  # T_n(-1) = (-1)^n
                    A[row, n+1] = Tn
                end
            end

        elseif bc_type == :neumann
            # du/dr(r_boundary) = 0
            # Use first derivative operator
            D1 = sparse_radial_operator(0, 1, N, ri, ro)
            A[row, :] = D1[row, :]

        # Additional BC types can be added here
        end
    end

    return nothing
end

end  # module UltrasphericalSpectral
