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
       apply_boundary_conditions!,
       chebyshev_coefficients,
       multiplication_matrix

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
# Spectral multiplication (Chebyshev coefficients and Mlam)
# -----------------------------------------------------------------------------

"""
    chebyshev_coefficients(power::Int, N::Int, ri::Real, ro::Real;
                          tol::Real=1e-9) -> Vector{Float64}

Compute the first N Chebyshev coefficients (from 0 to N-1) of the function
    r(x)^power where r(x) = ri + (ro-ri)*(x+1)/2
for x ∈ [-1,1].

This follows Kore's chebco function (utils.py:251-269).

For ricb=0, the Chebyshev domain [-1,1] is mapped to [-rcmb, rcmb].
For ricb>0, the Chebyshev domain [-1,1] is mapped to [ricb, rcmb].
"""
function chebyshev_coefficients(power::Int, N::Int, ri::Real, ro::Real;
                                tol::Real=1e-9)
    # Evaluate function at Chebyshev-Gauss points
    # x_i = cos(π(i+0.5)/N) for i = 0,...,N-1
    x = [cos(π * (i + 0.5) / N) for i in 0:N-1]

    # Map to physical domain
    if ri == 0
        # No inner core: map [-1,1] → [-ro, ro]
        r = ro .* x
    else
        # With inner core: map [-1,1] → [ri, ro]
        r = @. ri + (ro - ri) * (x + 1) / 2
    end

    # Evaluate r^power
    f_vals = r .^ power

    # Compute Chebyshev coefficients using DCT
    # This is the discrete cosine transform (DCT-II)
    # Using the definition: a_k = (2/N) * Σ f_i * cos(πk(i+0.5)/N)
    coeffs = zeros(N)
    for k in 0:N-1
        s = sum(f_vals[i+1] * cos(π * k * (i + 0.5) / N) for i in 0:N-1)
        coeffs[k+1] = (2.0 / N) * s
    end

    # First coefficient gets factor of 1/2
    coeffs[1] /= 2.0

    # Truncate small coefficients
    coeffs[abs.(coeffs) .<= tol] .= 0.0

    return coeffs
end

"""
    csl0(s, λ, j, k) -> Float64

Compute c_s^λ(j,k) using the formula from Kore (utils.py:920-940).
This is used in the Gegenbauer multiplication recurrence relation.
"""
function csl0(s::Int, λ::Real, j::Int, k::Int)
    if s > min(j, k)
        return 0.0
    end

    # Compute product using logarithms to avoid overflow
    p1 = 1.0
    p2 = 1.0
    p3 = (j + k + λ - 2s) / (j + k + λ - s)
    p4 = 1.0

    # Product from t=0 to λ-1
    for t in 0:(Int(λ)-1)
        p1 *= (k - s + λ - t) / (k - s + 1 + t)
        p2 *= (λ + t) / (1 + t)
    end

    # Product from t=0 to s-1
    for t in 0:(s-1)
        p3 *= (j - t) / (1 + t)
        p4 *= (k - s + 1 + t) / (k - s + λ + t)
    end

    return p1 * p2 * p3 * p4
end

"""
    csl(svec, λ, j, k) -> Vector{Float64}

Recursion for c_s^λ starting from c_svec[1]^λ(j,k).
Following Kore utils.py:944-958.
"""
function csl(svec::AbstractVector{Int}, λ::Real, j::Int, k::Int)
    out = zeros(length(svec))
    out[1] = csl0(svec[1], λ, j, k)

    k_running = k
    for i in 2:length(svec)
        s = svec[i-1]
        tmp1 = (j + k_running + λ - s) * (λ + s) * (j - s) *
               (2λ + j + k_running - s) * (k_running - s + λ)
        tmp2 = (j + k_running + λ - s + 1) * (s + 1) * (λ + j - s - 1) *
               (λ + j + k_running - s) * (k_running - s + 1)
        out[i] = out[i-1] * tmp1 / tmp2
        k_running += 2
    end

    return out
end

"""
    multiplication_matrix(a0::Vector{Float64}, λ::Real, N::Int;
                         vector_parity::Int=0) -> SparseMatrixCSC

Construct the multiplication matrix M such that if u = Σ a_n C_n^(λ)(x)
and we want to compute w = f(x) * u where f(x) = Σ b_k C_k^(λ)(x),
then the coefficients of w in the C^(λ) basis are given by M * a.

This implements Kore's Mlam function (utils.py:962-1059).

Parameters:
- a0: Chebyshev coefficients of the function to multiply by (in C^(λ) basis)
- λ: Gegenbauer order
- N: Size of the operator
- vector_parity: 0 for inner core (no parity optimization),
                 ±1 for no inner core (parity optimization)
"""
function multiplication_matrix(a0::Vector{Float64}, λ::Real, N::Int;
                              vector_parity::Int=0)
    # Check if coefficients are non-zero
    if sum(abs.(a0)) == 0
        return sparse(zeros(N, N))
    end

    # Find bandwidth
    nonzero_idx = findall(x -> x != 0, a0)
    bw = maximum(nonzero_idx) - 1  # 0-indexed

    # Extend coefficient vector
    a1 = zeros(2 * N)
    a1[1:N] = a0

    # Determine row and column ranges based on parity
    if vector_parity != 0
        # Determine parities
        last_nonzero = nonzero_idx[end] - 1  # 0-indexed
        rpower_parity = 1 - 2 * (last_nonzero % 2)
        lamb_parity = 1 - 2 * (Int(λ) % 2)
        operator_parity = rpower_parity * lamb_parity
        overall_parity = vector_parity * operator_parity

        # Row parity: j even when overall_parity = 1
        # Col parity: k even when vector_parity * lamb_parity = 1
        idj = (1 - overall_parity) ÷ 2
        idk = (1 - vector_parity * lamb_parity) ÷ 2
        jrange = idj:2:N-1  # 0-indexed
    else
        jrange = 0:N-1  # 0-indexed
        idk = 0
    end

    # Build multiplication matrix
    if λ > 0
        # Gegenbauer case: use recurrence relations
        rows = Int[]
        cols = Int[]
        vals = Float64[]

        for j in jrange
            k1 = max(0, j - bw - 1)
            k2 = min(N - 1, j + bw + 1)
            ka = k1:k2

            if vector_parity != 0
                # Filter columns by parity
                krange = [k for k in ka if k % 2 == idk]
            else
                krange = collect(ka)
            end

            for k in krange
                s0 = max(0, k - j)
                s = s0:k
                idx = 2 .* s .+ j .- k .+ 1  # Convert to 1-indexed
                a = a1[idx]

                # Compute Gegenbauer coefficients
                if s0 == 0
                    cvec = csl(collect(s), λ, k, j - k)
                elseif s0 == k - j
                    cvec = csl(collect(s), λ, k, k - j)
                else
                    cvec = csl(collect(s), λ, k, abs(j - k))
                end

                val = dot(a, cvec)
                if abs(val) > 1e-14
                    push!(rows, j + 1)  # Convert to 1-indexed
                    push!(cols, k + 1)
                    push!(vals, val)
                end
            end
        end

        return sparse(rows, cols, vals, N, N)

    else
        # Chebyshev case (λ = 0): use Toeplitz + Hankel
        # Following Kore utils.py lines 1036-1052
        a2 = copy(a0)
        a2[1] *= 2  # Double the first coefficient

        # Build Toeplitz matrix: T[i,j] = a2[|i-j|]
        # In 0-indexed: T[i,j] = a2[abs(i-j)]
        T = zeros(N, N)
        for i in 0:N-1
            for j in 0:N-1
                idx = abs(i - j) + 1  # Convert to 1-indexed
                if idx <= N
                    T[i+1, j+1] = a2[idx]
                end
            end
        end

        # Build Hankel matrix: H[i,j] = a2[i+j]
        # In 0-indexed: H[i,j] = a2[i+j]
        H = zeros(N, N)
        for i in 0:N-1
            for j in 0:N-1
                idx = i + j + 1  # Convert to 1-indexed
                if idx <= N
                    H[i+1, j+1] = a2[idx]
                end
            end
        end

        # Set first row of Hankel to zero (Kore line 1040)
        H[1, :] .= 0.0

        # Combine: out = 0.5 * (Toeplitz + Hankel)
        tmp = 0.5 * (T + H)

        # Apply parity constraints
        if vector_parity != 0
            idj0 = (1 + overall_parity) ÷ 2
            idk0 = (1 + vector_parity * lamb_parity) ÷ 2
            for j in idj0:2:N-1
                tmp[j+1, :] .= 0.0
            end
            for k in idk0:2:N-1
                tmp[:, k+1] .= 0.0
            end
        end

        return sparse(tmp)
    end
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

    # Apply r^power multiplication using SPECTRAL multiplication (not physical!)
    if power != 0
        # Step 1: Get Chebyshev coefficients of r^power
        # Need N+1 coefficients to match matrix size
        r_coeffs = chebyshev_coefficients(power, N+1, ri, ro)

        # Step 2: Convert r^power coefficients to current Gegenbauer basis C^(λ)
        # If λ = 0, r_coeffs are already in the right basis
        # If λ > 0, we need to convert through the ultraspherical chain
        r_coeffs_lambda = r_coeffs
        for lam in 0:(Int(λ)-1)
            S = ultraspherical_conversion(lam, N)
            r_coeffs_lambda = S * r_coeffs_lambda
        end

        # Step 3: Build multiplication matrix in C^(λ) basis
        # vector_parity = 0 for inner core case (ricb > 0)
        # This is the proper spectral multiplication operator
        M = multiplication_matrix(r_coeffs_lambda, λ, N+1; vector_parity=0)

        # Step 4: Apply multiplication: D_new = M * D
        D = M * D
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
  - :neumann2 → d²u/dr² = 0 (for stress-free)
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
            if row == 1 || row == 2  # Outer boundary (x = 1, r = ro)
                for n in 0:N
                    Tn = cos(n * 0.0)  # T_n(1) = 1 for all n
                    A[row, n+1] = Tn
                end
            elseif row == N || row == N+1  # Inner boundary (x = -1, r = ri)
                for n in 0:N
                    Tn = cos(n * π)  # T_n(-1) = (-1)^n
                    A[row, n+1] = Tn
                end
            end

        elseif bc_type == :neumann
            # du/dr(r_boundary) = 0
            # Use first derivative operator
            D1 = sparse_radial_operator(0, 1, N, ri, ro)
            # Extract the appropriate boundary row
            if row == 1 || row == 2  # Outer boundary
                A[row, 1:(N+1)] = D1[1, :]
            elseif row == N || row == N+1  # Inner boundary
                A[row, 1:(N+1)] = D1[N+1, :]
            end

        elseif bc_type == :neumann2
            # d²u/dr²(r_boundary) = 0
            # Use second derivative operator
            D2 = sparse_radial_operator(0, 2, N, ri, ro)
            # Extract the appropriate boundary row
            if row == 1 || row == 2  # Outer boundary
                A[row, 1:(N+1)] = D2[1, :]
            elseif row == N || row == N+1  # Inner boundary
                A[row, 1:(N+1)] = D2[N+1, :]
            end

        # Additional BC types can be added here
        end
    end

    return nothing
end

end  # module UltrasphericalSpectral
