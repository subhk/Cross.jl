# =============================================================================
#  Boundary-condition utilities in toroidal–poloidal representation
# =============================================================================

"""
    velocity_from_potentials(op, P, T)

Convert poloidal (`P`) and toroidal (`T`) potentials defined on the operator
collocation grid into velocity components `(u_r, u_θ, u_φ)` for the azimuthal
wavenumber `op.params.m`.

The formulas follow the standard decomposition

```
    u = ∇ × ∇ × (P r̂) + ∇ × (T r̂)
```

assuming fields vary as `exp(i m φ)`.  The returned arrays share the same shape
as the input potentials.
"""
function velocity_from_potentials(op, P, T)
    # Angular derivatives
    dθ_P = P * op.Dθ'
    dθ_T = T * op.Dθ'
    lap_ang_P = P * op.Lθ'

    # Radial derivatives of the potentials
    dr_P = op.Dr * P

    # Common geometric factors
    inv_r = op.inv_r
    inv_r2 = op.inv_r2
    inv_r_sinθ = op.inv_r_sinθ

    im_m = op.im_m

    # Velocity components
    u_r = -lap_ang_P .* inv_r2
    u_θ = (dr_P * op.Dθ') .* inv_r .+ (im_m .* T) .* inv_r_sinθ
    u_φ = (im_m .* dr_P) .* inv_r_sinθ .- (dθ_T .* inv_r)

    return u_r, u_θ, u_φ
end

"""
    apply_mechanical_bc_from_potentials!(res_r, res_θ, res_φ,
                                         P, T, op;
                                         inner::Symbol=:no_slip,
                                         outer::Symbol=:no_slip)

Overwrite the boundary rows of the residual blocks `(res_r, res_θ, res_φ)` using
velocity boundary conditions derived from the toroidal–poloidal potentials
`(P, T)`.

Supported mechanical boundary types:

- `:no_slip`      → `u_r = u_θ = u_φ = 0`
- `:stress_free`  → `u_r = 0`, `∂_r u_θ = u_θ / r`, `∂_r u_φ = u_φ / r`

The function evaluates the necessary velocity components (and their radial
derivatives) internally from the potentials.
"""
function apply_mechanical_bc_from_potentials!(res_r, res_θ, res_φ,
                                              P, T, op;
                                              inner::Symbol=:no_slip,
                                              outer::Symbol=:no_slip)
    u_r, u_θ, u_φ = velocity_from_potentials(op, P, T)
    dr_uθ = op.Dr * u_θ
    dr_uφ = op.Dr * u_φ

    enforce_mechanical_bc_at!(res_r, res_θ, res_φ,
                              u_r, u_θ, u_φ,
                              dr_uθ, dr_uφ,
                              op, inner, 1)

    enforce_mechanical_bc_at!(res_r, res_θ, res_φ,
                              u_r, u_θ, u_φ,
                              dr_uθ, dr_uφ,
                              op, outer, op.Nr)
    return nothing
end

function enforce_mechanical_bc_at!(res_r, res_θ, res_φ,
                                   u_r, u_θ, u_φ,
                                   dr_uθ, dr_uφ,
                                   op, bc::Symbol, idx::Int)
    if bc === :no_slip
        res_r[idx, :] .= u_r[idx, :]
        res_θ[idx, :] .= u_θ[idx, :]
        res_φ[idx, :] .= u_φ[idx, :]
    elseif bc === :stress_free
        res_r[idx, :] .= u_r[idx, :]
        res_θ[idx, :] .= dr_uθ[idx, :] .- u_θ[idx, :] .* op.inv_r[idx, :]
        res_φ[idx, :] .= dr_uφ[idx, :] .- u_φ[idx, :] .* op.inv_r[idx, :]
    else
        throw(ArgumentError("Unsupported mechanical boundary condition: $(bc)"))
    end
end

"""
    apply_thermal_bc_from_potentials!(res_T, Θ, op;
                                      inner::Symbol=:fixed_temperature,
                                      outer::Symbol=:fixed_temperature,
                                      value_inner::Real=0.0,
                                      value_outer::Real=0.0,
                                      flux_inner::Real=0.0,
                                      flux_outer::Real=0.0)

Apply thermal boundary conditions directly to the temperature residual block
`res_T`.  The helper mirrors the mechanical routine but does not require
potentials explicitly; it is defined here so that a single module hosts all
boundary utilities for the toroidal–poloidal formulation.

Supported thermal boundary types:

- `:fixed_temperature` → Θ = prescribed value
- `:fixed_flux`        → ∂_r Θ = prescribed flux
"""
function apply_thermal_bc_from_potentials!(res_T, Θ, op;
                                           inner::Symbol=:fixed_temperature,
                                           outer::Symbol=:fixed_temperature,
                                           value_inner::Real=0.0,
                                           value_outer::Real=0.0,
                                           flux_inner::Real=0.0,
                                           flux_outer::Real=0.0)
    dΘ_dr = op.Dr * Θ
    apply_thermal_bc_at!(res_T, Θ, dΘ_dr, inner, value_inner, flux_inner, 1)
    apply_thermal_bc_at!(res_T, Θ, dΘ_dr, outer, value_outer, flux_outer, op.Nr)
    return nothing
end

function apply_thermal_bc_at!(res_T, Θ, dΘ_dr,
                              bc::Symbol,
                              value::Real,
                              flux::Real,
                              idx::Int)
    if bc === :fixed_temperature
        res_T[idx, :] .= Θ[idx, :] .- value
    elseif bc === :fixed_flux
        res_T[idx, :] .= dΘ_dr[idx, :] .- flux
    else
        throw(ArgumentError("Unsupported thermal boundary condition: $(bc)"))
    end
end
