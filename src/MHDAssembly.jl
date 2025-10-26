# =============================================================================
#  MHD Matrix Assembly
#
#  Assembles the full MHD generalized eigenvalue problem:
#      A * x = λ * B * x
#
#  Where x = [u, v, f, g, h] contains:
#  - u: poloidal velocity perturbation
#  - v: toroidal velocity perturbation
#  - f: poloidal magnetic field perturbation
#  - g: toroidal magnetic field perturbation
#  - h: temperature perturbation
#
#  Following Kore's assemble.py structure
# =============================================================================

"""
Module for MHD matrix assembly.
Must be included after MHDOperator.jl and MHDOperatorFunctions.jl
"""

# This file is meant to be included in MHDOperator.jl

"""
    assemble_mhd_matrices(op::MHDStabilityOperator)

Assemble the full MHD matrices A and B for the generalized eigenvalue problem.

Returns: (A, B, interior_dofs, info)

# Matrix Structure

The matrix is organized in 5 sections:
1. Section u (poloidal velocity): 2curl Navier-Stokes with Lorentz force
2. Section v (toroidal velocity): 1curl Navier-Stokes with Lorentz force
3. Section f (poloidal B field): no-curl induction equation
4. Section g (toroidal B field): 1curl induction equation
5. Section h (temperature): heat equation with advection

# Couplings

Velocity → Velocity: Coriolis, viscous diffusion
Velocity → Magnetic: Induction (u,v → f,g)
Magnetic → Velocity: Lorentz force (f,g → u,v)
Magnetic → Magnetic: Magnetic diffusion
Velocity → Temperature: Thermal advection
Temperature → Velocity: Buoyancy
"""
function assemble_mhd_matrices(op::MHDStabilityOperator{T}) where {T}
    params = op.params
    E = params.E
    Pr = params.Pr
    Pm = params.Pm
    Ra = params.Ra
    Le = params.Le
    Etherm = params.Etherm
    Em = params.Em
    m = params.m
    N = params.N
    ricb = params.ricb

    n = op.matrix_size
    n_per_mode = N + 1

    nb_u = length(op.ll_u)
    nb_v = length(op.ll_v)
    nb_f = length(op.ll_f)
    nb_g = length(op.ll_g)
    nb_h = length(op.ll_h)

    println("\nAssembling MHD sparse matrices...")
    println("  Matrix size: $n × $n")
    println("  Sections: u($nb_u modes), v($nb_v modes), f($nb_f modes), g($nb_g modes), h($nb_h modes)")

    # Use COO format for efficient assembly
    A_rows = Int[]
    A_cols = Int[]
    A_vals = Float64[]
    B_rows = Int[]
    B_cols = Int[]
    B_vals = Float64[]

    # Helper function to add block to sparse matrix
    function add_block!(rows, cols, vals, block, row_offset, col_offset)
        Is, Js, Vs = findnz(block)
        append!(rows, Is .+ row_offset)
        append!(cols, Js .+ col_offset)
        append!(vals, Vs)
    end

    # =========================================================================
    # SECTION U: Poloidal Velocity (2curl Navier-Stokes + Lorentz)
    # =========================================================================
    println("  Assembling section u (poloidal velocity)...")

    for (k, l) in enumerate(op.ll_u)
        row_base = (k - 1) * n_per_mode
        col_base = (k - 1) * n_per_mode

        L = l * (l + 1)

        # ---------------------------------------------------------------------
        # B matrix: Time derivative (inertia)
        # ---------------------------------------------------------------------
        u_op = operator_u(op, l)
        add_block!(B_rows, B_cols, B_vals, u_op, row_base, col_base)

        # ---------------------------------------------------------------------
        # A matrix: RHS operators
        # ---------------------------------------------------------------------

        # Coriolis force (diagonal)
        cori_op = operator_coriolis_diagonal(op, l, m)
        add_block!(A_rows, A_cols, A_vals, cori_op, row_base, col_base)

        # Viscous diffusion
        visc_op = -operator_viscous_diffusion(op, l, E)
        add_block!(A_rows, A_cols, A_vals, visc_op, row_base, col_base)

        # Buoyancy (coupling from temperature)
        if Ra > 0
            buoy_op = -operator_buoyancy(op, l, Ra, Pr)
            # Column offset for temperature section
            temp_col_base = (nb_u + nb_v + nb_f + nb_g + k - 1) * n_per_mode
            add_block!(A_rows, A_cols, A_vals, buoy_op, row_base, temp_col_base)
        end

        # Lorentz force from magnetic field (if Le > 0)
        if Le > 0
            # Diagonal: toroidal B at same l
            lorentz_diag = operator_lorentz_poloidal_diagonal(op, l, Le)
            g_col_base = (nb_u + nb_v + nb_f + k - 1) * n_per_mode
            add_block!(A_rows, A_cols, A_vals, lorentz_diag, row_base, g_col_base)

            # Off-diagonal: toroidal B at l±1
            for offset in [-1, 1]
                l_coupled = l + offset
                if l_coupled in op.ll_g
                    k_coupled = findfirst(==(l_coupled), op.ll_g)
                    g_col_coupled = (nb_u + nb_v + nb_f + k_coupled - 1) * n_per_mode

                    lorentz_off = operator_lorentz_poloidal_offdiag(op, l, m, offset, Le)
                    add_block!(A_rows, A_cols, A_vals, lorentz_off, row_base, g_col_coupled)
                end
            end
        end

        # Coriolis off-diagonal: u ↔ v coupling
        for offset in [-1, 1]
            l_coupled = l + offset
            if l_coupled in op.ll_v
                k_coupled = findfirst(==(l_coupled), op.ll_v)
                v_col_coupled = (nb_u + k_coupled - 1) * n_per_mode

                cori_off, _ = operator_coriolis_offdiag(op, l, m, offset)
                add_block!(A_rows, A_cols, A_vals, cori_off, row_base, v_col_coupled)
            end
        end
    end

    # =========================================================================
    # SECTION V: Toroidal Velocity (1curl Navier-Stokes + Lorentz)
    # =========================================================================
    println("  Assembling section v (toroidal velocity)...")

    for (k, l) in enumerate(op.ll_v)
        row_base = (nb_u + k - 1) * n_per_mode
        col_base = (nb_u + k - 1) * n_per_mode

        # ---------------------------------------------------------------------
        # B matrix: Time derivative
        # ---------------------------------------------------------------------
        v_op = operator_u_toroidal(op, l)
        add_block!(B_rows, B_cols, B_vals, v_op, row_base, col_base)

        # ---------------------------------------------------------------------
        # A matrix: RHS operators
        # ---------------------------------------------------------------------

        # Coriolis (diagonal)
        cori_tor = operator_coriolis_toroidal(op, l, m)
        add_block!(A_rows, A_cols, A_vals, cori_tor, row_base, col_base)

        # Viscous diffusion
        visc_tor = -operator_viscous_toroidal(op, l, E)
        add_block!(A_rows, A_cols, A_vals, visc_tor, row_base, col_base)

        # Lorentz force from poloidal B (if Le > 0)
        if Le > 0
            lorentz_tor = operator_lorentz_toroidal(op, l, Le)
            f_col_base = (nb_u + nb_v + k - 1) * n_per_mode
            add_block!(A_rows, A_cols, A_vals, lorentz_tor, row_base, f_col_base)
        end

        # Coriolis reverse coupling: v → u at l±1
        for offset in [-1, 1]
            l_coupled = l + offset
            if l_coupled in op.ll_u
                k_coupled = findfirst(==(l_coupled), op.ll_u)
                u_col_coupled = (k_coupled - 1) * n_per_mode

                cori_v_to_u = operator_coriolis_v_to_u(op, l, m, offset)
                add_block!(A_rows, A_cols, A_vals, cori_v_to_u, row_base, u_col_coupled)
            end
        end
    end

    # =========================================================================
    # SECTION F: Poloidal Magnetic Field (no-curl induction)
    # =========================================================================
    println("  Assembling section f (poloidal magnetic field)...")

    for (k, l) in enumerate(op.ll_f)
        row_base = (nb_u + nb_v + k - 1) * n_per_mode
        col_base = (nb_u + nb_v + k - 1) * n_per_mode

        # ---------------------------------------------------------------------
        # B matrix: Time derivative
        # ---------------------------------------------------------------------
        b_pol = operator_b_poloidal(op, l)
        add_block!(B_rows, B_cols, B_vals, b_pol, row_base, col_base)

        # ---------------------------------------------------------------------
        # A matrix: RHS operators
        # ---------------------------------------------------------------------

        # Magnetic diffusion
        mag_diff_pol = operator_magnetic_diffusion_poloidal(op, l, Em)
        add_block!(A_rows, A_cols, A_vals, mag_diff_pol, row_base, col_base)

        # Induction from poloidal velocity u
        if Le > 0  # Only if there's a background field
            u_col_base = (k - 1) * n_per_mode
            induct_from_u = operator_induction_poloidal_from_u(op, l)
            add_block!(A_rows, A_cols, A_vals, induct_from_u, row_base, u_col_base)

            # Induction from toroidal velocity v (if l exists in v)
            if l in op.ll_v
                k_v = findfirst(==(l), op.ll_v)
                v_col_base = (nb_u + k_v - 1) * n_per_mode
                induct_from_v = operator_induction_poloidal_from_v(op, l)
                add_block!(A_rows, A_cols, A_vals, induct_from_v, row_base, v_col_base)
            end
        end
    end

    # =========================================================================
    # SECTION G: Toroidal Magnetic Field (1curl induction)
    # =========================================================================
    println("  Assembling section g (toroidal magnetic field)...")

    for (k, l) in enumerate(op.ll_g)
        row_base = (nb_u + nb_v + nb_f + k - 1) * n_per_mode
        col_base = (nb_u + nb_v + nb_f + k - 1) * n_per_mode

        # ---------------------------------------------------------------------
        # B matrix: Time derivative
        # ---------------------------------------------------------------------
        b_tor = operator_b_toroidal(op, l)
        add_block!(B_rows, B_cols, B_vals, b_tor, row_base, col_base)

        # ---------------------------------------------------------------------
        # A matrix: RHS operators
        # ---------------------------------------------------------------------

        # Magnetic diffusion
        mag_diff_tor = operator_magnetic_diffusion_toroidal(op, l, Em)
        add_block!(A_rows, A_cols, A_vals, mag_diff_tor, row_base, col_base)

        # Induction from velocity field (if Le > 0)
        if Le > 0
            # From toroidal velocity v (diagonal)
            if l in op.ll_v
                k_v = findfirst(==(l), op.ll_v)
                v_col_base = (nb_u + k_v - 1) * n_per_mode
                induct_v_tor = operator_induction_toroidal_from_v(op, l)
                add_block!(A_rows, A_cols, A_vals, induct_v_tor, row_base, v_col_base)
            end

            # From poloidal velocity u (off-diagonal l±1)
            for offset in [-1, 1]
                l_coupled = l + offset
                if l_coupled in op.ll_u
                    k_coupled = findfirst(==(l_coupled), op.ll_u)
                    u_col_coupled = (k_coupled - 1) * n_per_mode

                    induct_u_tor = operator_induction_toroidal_from_u(op, l, m, offset)
                    add_block!(A_rows, A_cols, A_vals, induct_u_tor, row_base, u_col_coupled)
                end
            end
        end
    end

    # =========================================================================
    # SECTION H: Temperature (same as hydrodynamic case)
    # =========================================================================
    println("  Assembling section h (temperature)...")

    for (k, l) in enumerate(op.ll_h)
        row_base = (nb_u + nb_v + nb_f + nb_g + k - 1) * n_per_mode
        col_base = (nb_u + nb_v + nb_f + nb_g + k - 1) * n_per_mode

        # ---------------------------------------------------------------------
        # B matrix: Time derivative
        # ---------------------------------------------------------------------
        theta_op = operator_theta(op, l)
        add_block!(B_rows, B_cols, B_vals, theta_op, row_base, col_base)

        # ---------------------------------------------------------------------
        # A matrix: RHS operators
        # ---------------------------------------------------------------------

        # Thermal diffusion
        thermal_diff = operator_thermal_diffusion(op, l, Etherm)
        add_block!(A_rows, A_cols, A_vals, thermal_diff, row_base, col_base)

        # Thermal advection (from poloidal velocity)
        vel_col_base = (k - 1) * n_per_mode
        thermal_adv = operator_thermal_advection(op, l)
        add_block!(A_rows, A_cols, A_vals, thermal_adv, row_base, vel_col_base)
    end

    # =========================================================================
    # Convert to sparse CSC format
    # =========================================================================
    println("  Converting to CSC format...")
    A = sparse(A_rows, A_cols, A_vals, n, n)
    B = sparse(B_rows, B_cols, B_vals, n, n)

    println("  A sparsity: $(nnz(A)) / $(n^2) = $(100*nnz(A)/n^2)%")
    println("  B sparsity: $(nnz(B)) / $(n^2) = $(100*nnz(B)/n^2)%")

    # =========================================================================
    # Apply boundary conditions
    # =========================================================================
    println("  Applying boundary conditions...")

    # Velocity BCs (same as hydrodynamic)
    apply_velocity_boundary_conditions!(A, B, op)

    # Magnetic field BCs
    apply_magnetic_boundary_conditions!(A, B, op, :f)
    apply_magnetic_boundary_conditions!(A, B, op, :g)

    # Temperature BCs (same as hydrodynamic)
    apply_temperature_boundary_conditions!(A, B, op)

    println("  Final A sparsity: $(nnz(A)) / $(n^2)")
    println("  Final B sparsity: $(nnz(B)) / $(n^2)")

    # Identify interior DOFs
    B_diag = diag(B)
    interior_dofs = findall(i -> abs(B_diag[i]) > 1e-14, 1:n)
    println("  Interior DOFs: $(length(interior_dofs)) / $n")

    info = Dict(
        "method" => "MHD sparse ultraspherical",
        "N" => N,
        "lmax" => params.lmax,
        "m" => m,
        "nl_modes" => op.nl_modes,
        "matrix_size" => n,
        "sections" => "u, v, f, g, h"
    )

    return A, B, interior_dofs, info
end

# -----------------------------------------------------------------------------
# Boundary condition helpers
# -----------------------------------------------------------------------------

function apply_velocity_boundary_conditions!(A, B, op)
    # Same as hydrodynamic case - reuse from SparseOperator
    # (implementation omitted for brevity - copy from SparseOperator.jl)
end

function apply_temperature_boundary_conditions!(A, B, op)
    # Same as hydrodynamic case - reuse from SparseOperator
    # (implementation omitted for brevity - copy from SparseOperator.jl)
end
