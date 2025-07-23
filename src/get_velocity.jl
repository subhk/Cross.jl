# =============================================================================
#  Convert toroidal/poloidal potentials to velocity components (u_r,u_θ,u_φ)
# =============================================================================
"""
    potentials_to_velocity(P, T; Dr, Dθ, Lθ, r, sinθ, m)

Given poloidal `P` and toroidal `T` potentials on the (r,θ) grid
(size N_r×N_θ) return three Complex arrays `(ur, uθ, uφ)` containing the
velocity components for the Fourier mode `m` (fields vary like `exp(i m φ)`).

Uses the same derivative matrices `Dr`, `Dθ` and angular operator `Lθ`
constructed earlier in the script.
"""
function potentials_to_velocity(P::AbstractMatrix,
                                T::AbstractMatrix;
                                Dr, Dθ, Lθ, r::AbstractVector,
                                sinθ::AbstractVector, m::Int)

    N_r, N_θ = size(P)
    @assert size(T) == (N_r, N_θ)

    # allocate outputs
    ur  = similar(P, ComplexF64)
    uθ  = similar(P, ComplexF64)
    uφ  = similar(P, ComplexF64)

    # precompute 1/r² and 1/sinθ vectors
    inv_r2 = 1.0 ./ (r.^2)
    inv_s  = 1.0 ./ sinθ

    # latitudinal derivatives: ∂θ P and ∂θ T
    ∂θP = P * Dθ'          # (N_r×N_θ) * (N_θ×N_θ)
    ∂θT = T * Dθ'

    # angular Laplacian Lθ P
    LθP = P * Lθ'

    # radial derivative of P and T
    ∂rP = Dr * P           # (N_r×N_r)*(N_r×N_θ)
    # ∂r of (im m / sinθ * P) = im m / sinθ * ∂rP  (sinθ independent of r)

    im_m = im * m

    # loop over θ to apply 1/sinθ factors cheaply
    for j in 1:N_θ
        s⁻¹ = inv_s[j]
        for i in 1:N_r
            # u_r
            ur[i,j] = LθP[i,j] * inv_r2[i]

            # u_θ
            uθ[i,j] = ∂rP[i,j] * Dθ'[j,j]  # will overwrite below
        end
    end

    # Correct uθ: ∂r∂θ P = (∂rP) * Dθ'
    uθ .= (∂rP * Dθ') .- (im_m .* T) .* (ones(N_r) * inv_s')

    # uφ
    uφ .= (im_m .* ∂rP) .* (ones(N_r) * inv_s') .+ ∂θT

    return ur, uθ, uφ
end

"""
# --- Example usage on the neutral eigen‑mode just computed -------------------
ur_mode, uθ_mode, uφ_mode =
    potentials_to_velocity(P_mode, T_mode;
                           Dr=Dr, Dθ=Dθ, Lθ=Lθ, r=r, sinθ=sinθ, m=m)

@printf("  ||u_r||₂ = %.3e   ||u_θ||₂ = %.3e   ||u_φ||₂ = %.3e\n",
        norm(ur_mode), norm(uθ_mode), norm(uφ_mode))
"""