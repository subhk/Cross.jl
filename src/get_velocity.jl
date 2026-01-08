# =============================================================================
#  Reconstruction of velocity and temperature fields from spectral coefficients
#
#  Provides functions to convert poloidal/toroidal potentials to physical
#  velocity components on a meridional (r, θ) grid for a single azimuthal mode.
# =============================================================================

using LinearAlgebra

"""
    potentials_to_velocity(P, T; Dr, Dθ, Lθ, r, sintheta, m)

Compute velocity components `(u_r, u_θ, u_φ)` from poloidal and toroidal
potentials on a meridional (r, θ) grid for a single azimuthal mode m.

The velocity is reconstructed using the poloidal-toroidal decomposition:

    u = ∇×∇×(P r̂) + ∇×(T r̂)

which gives:
    u_r = -L²P / r²
    u_θ = (1/r) ∂(∂P/∂r)/∂θ + (im/r sinθ) T
    u_φ = (im/r sinθ) ∂P/∂r - (1/r) ∂T/∂θ

where L² = -ℓ(ℓ+1) is the angular Laplacian eigenvalue.

# Arguments
- `P::AbstractMatrix` - Poloidal potential on (Nr, Nθ) grid
- `T::AbstractMatrix` - Toroidal potential on (Nr, Nθ) grid
- `Dr` - Radial differentiation matrix (Nr × Nr)
- `Dθ` - Colatitude differentiation matrix (Nθ × Nθ)
- `Lθ` - Angular Laplacian operator (Nθ × Nθ)
- `r::AbstractVector` - Radial coordinates (length Nr)
- `sintheta::AbstractVector` - sin(θ) values (length Nθ)
- `m::Int` - Azimuthal wavenumber

# Returns
- `(u_r, u_θ, u_φ)` - Velocity components as (Nr, Nθ) complex matrices

# Example
```julia
# Set up grid and operators
Nr, Nθ = 64, 128
cd = ChebyshevDiffn(Nr, [χ, 1.0], 2)
Dr = cd.D1
# ... set up Dθ, Lθ, sintheta ...

# Reconstruct velocity from eigenvector
u_r, u_θ, u_φ = potentials_to_velocity(P, T; Dr=Dr, Dθ=Dθ, Lθ=Lθ,
                                        r=cd.x, sintheta=sintheta, m=m)
```
"""
function potentials_to_velocity(P::AbstractMatrix,
                                T::AbstractMatrix;
                                Dr,
                                Dθ,
                                Lθ,
                                r::AbstractVector,
                                sintheta::AbstractVector,
                                m::Int)
    Nr, Nθ = size(P)
    size(T) == size(P) || throw(DimensionMismatch("P and T must have same size"))
    @assert size(Dr, 1) == Nr
    @assert size(Dθ, 1) == Nθ
    @assert length(r) == Nr
    @assert length(sintheta) == Nθ

    inv_r = 1.0 ./ r
    inv_r2 = inv_r .^ 2
    inv_sinθ = 1.0 ./ sintheta

    dθ_P = P * Dθ'
    dθ_T = T * Dθ'
    lap_ang_P = P * Lθ'
    dP_dr = Dr * P

    # u_r = -L²P / r² (L² applied via Lθ operator)
    ur = -lap_ang_P .* inv_r2

    # u_θ = (1/r) ∂²P/∂r∂θ + (im/r sinθ) T
    uθ = dP_dr * Dθ'
    uθ .= uθ .* inv_r .* (ones(Nr) * ones(Nθ)')
    uθ .+= (im * m) .* T .* (inv_r * inv_sinθ')

    # u_φ = (im/r sinθ) ∂P/∂r - (1/r) ∂T/∂θ
    uφ = (im * m) .* dP_dr .* (inv_r * inv_sinθ')
    uφ .-= dθ_T .* (inv_r * ones(1, Nθ))

    return ur, uθ, uφ
end
