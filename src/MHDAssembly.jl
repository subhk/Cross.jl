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
# =============================================================================

"""
Module for MHD matrix assembly.
Must be included after MHDOperator.jl and MHDOperatorFunctions.jl
"""

# This file is meant to be included in CompleteMHD.jl

# Import operator functions from SparseOperator so we can extend them
import .SparseOperator: operator_u, operator_coriolis_diagonal, operator_coriolis_offdiag,
                        operator_viscous_diffusion, operator_buoyancy, operator_coriolis_v_to_u,
                        operator_u_toroidal, operator_coriolis_toroidal, operator_viscous_toroidal,
                        operator_theta, operator_thermal_diffusion, operator_thermal_advection
import .MHDOperator: is_dipole_case

# -----------------------------------------------------------------------------
# Inline operator construction for MHDStabilityOperator
# These extend the functions from SparseOperator.jl to work with MHDStabilityOperator
# -----------------------------------------------------------------------------

function operator_u(op::MHDStabilityOperator{T}, l::Int) where {T}
    L = l * (l + 1)
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return L * (L * op.r4_D0_u - 2 * op.r5_D1_u - op.r6_D2_u)
    else
        return L * (L * op.r2_D0_u - 2 * op.r3_D1_u - op.r4_D2_u)
    end
end

function operator_coriolis_diagonal(op::MHDStabilityOperator{T}, l::Int, m::Int) where {T}
    L = l * (l + 1)
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return 2im * m * (-L * op.r4_D0_u + 2 * op.r5_D1_u + op.r6_D2_u)
    else
        return 2im * m * (-L * op.r2_D0_u + 2 * op.r3_D1_u + op.r4_D2_u)
    end
end

function operator_coriolis_offdiag(op::MHDStabilityOperator{T}, l::Int, m::Int, offset::Int) where {T}
    dipole = is_dipole_case(op.params.B0_type, op.params.ricb)
    if offset == -1
        C = (l^2 - 1) * sqrt(l^2 - m^2) / (2l - 1)
        if dipole
            mtx = 2 * C * ((l - 1) * op.r5_D0_u - op.r6_D1_u)
        else
            mtx = 2 * C * ((l - 1) * op.r3_D0_u - op.r4_D1_u)
        end
        return mtx, -1
    elseif offset == 1
        C = l * (l + 2) * sqrt((l + m + 1) * (l - m + 1)) / (2l + 3)
        if dipole
            mtx = 2 * C * (-(l + 2) * op.r5_D0_u - op.r6_D1_u)
        else
            mtx = 2 * C * (-(l + 2) * op.r3_D0_u - op.r4_D1_u)
        end
        return mtx, 1
    else
        error("offset must be ±1 for Coriolis off-diagonal")
    end
end

function operator_viscous_diffusion(op::MHDStabilityOperator{T}, l::Int, E::T) where {T}
    L = l * (l + 1)
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return E * L * (-L * (l + 2) * (l - 1) * op.r2_D0_u +
                        2 * L * op.r4_D2_u -
                        4 * op.r5_D3_u -
                        op.r6_D4_u)
    else
        return E * L * (-L * (l + 2) * (l - 1) * op.r0_D0_u +
                        2 * L * op.r2_D2_u -
                        4 * op.r3_D3_u -
                        op.r4_D4_u)
    end
end

function operator_buoyancy(op::MHDStabilityOperator{T}, l::Int, Ra::T, Pr::T) where {T}
    # Convert gap-based Ra to internal Ra
    # Ra_internal = Ra_gap / gap^3 (gap = r_o - r_i = 1 - ricb when r_o = 1)
    E = op.params.E
    ricb = op.params.ricb
    gap = one(T) - ricb
    Ra_internal = Ra / gap^3

    # Beyonce factor = BV² = -Ra_internal * E² / Pr
    beyonce = -Ra_internal * E^2 / Pr
    L = l * (l + 1)
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return beyonce * L * op.r6_D0_u
    else
        return beyonce * L * op.r4_D0_u
    end
end

function operator_coriolis_v_to_u(op::MHDStabilityOperator{T}, l::Int, m::Int, offset::Int) where {T}
    dipole = is_dipole_case(op.params.B0_type, op.params.ricb)
    if offset == -1
        C = (l^2 - 1) * sqrt(l^2 - m^2) / (2l - 1)
        if dipole
            return 2 * C * ((l - 1) * op.r4_D0_v - op.r5_D1_v)
        else
            return 2 * C * ((l - 1) * op.r1_D0_v - op.r2_D1_v)
        end
    elseif offset == 1
        C = l * (l + 2) * sqrt((l + m + 1) * (l - m + 1)) / (2l + 3)
        if dipole
            return 2 * C * (-(l + 2) * op.r4_D0_v - op.r5_D1_v)
        else
            return 2 * C * (-(l + 2) * op.r1_D0_v - op.r2_D1_v)
        end
    else
        error("offset must be ±1 for Coriolis v→u coupling")
    end
end

function operator_u_toroidal(op::MHDStabilityOperator{T}, l::Int) where {T}
    L = l * (l + 1)
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return L * op.r5_D0_v
    else
        return L * op.r2_D0_v
    end
end

function operator_coriolis_toroidal(op::MHDStabilityOperator{T}, l::Int, m::Int) where {T}
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return -2im * m * op.r5_D0_v
    else
        return -2im * m * op.r2_D0_v
    end
end

function operator_viscous_toroidal(op::MHDStabilityOperator{T}, l::Int, E::T) where {T}
    L = l * (l + 1)
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return E * L * (-L * op.r3_D0_v + 2 * op.r4_D1_v + op.r5_D2_v)
    else
        return E * L * (-L * op.r0_D0_v + 2 * op.r1_D1_v + op.r2_D2_v)
    end
end

function operator_theta(op::MHDStabilityOperator{T}, l::Int) where {T}
    if op.params.heating == :differential
        return op.r3_D0_h
    else  # :internal
        return op.r2_D0_h
    end
end

function operator_thermal_diffusion(op::MHDStabilityOperator{T}, l::Int, Etherm::T) where {T}
    L = l * (l + 1)
    if op.params.heating == :differential
        return Etherm * (-L * op.r1_D0_h + 2 * op.r2_D1_h + op.r3_D2_h)
    else  # :internal
        return Etherm * (-L * op.r0_D0_h + 2 * op.r1_D1_h + op.r2_D2_h)
    end
end

function operator_thermal_advection(op::MHDStabilityOperator{T}, l::Int) where {T}
    L = l * (l + 1)
    if op.params.heating == :differential
        ricb = op.params.ricb
        gap = one(T) - ricb
        return L * op.r0_D0_h * (ricb / gap)
    else  # :internal
        return L * op.r2_D0_h
    end
end

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
    section_info = String[]
    nb_u > 0 && push!(section_info, "u($nb_u modes)")
    nb_v > 0 && push!(section_info, "v($nb_v modes)")
    nb_f > 0 && push!(section_info, "f($nb_f modes)")
    nb_g > 0 && push!(section_info, "g($nb_g modes)")
    nb_h > 0 && push!(section_info, "h($nb_h modes)")
    sections_text = join(section_info, ", ")
    println("  Sections: $sections_text")

    # Use COO format for efficient assembly
    # Use ComplexF64 because Coriolis operator has imaginary terms
    A_rows = Int[]
    A_cols = Int[]
    A_vals = ComplexF64[]
    B_rows = Int[]
    B_cols = Int[]
    B_vals = ComplexF64[]

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

        # ---------------------------------------------------------------------
        # B matrix: Time derivative (inertia)
        # ---------------------------------------------------------------------
        u_op = operator_u(op, l)
        add_block!(B_rows, B_cols, B_vals, -u_op, row_base, col_base)

        # ---------------------------------------------------------------------
        # A matrix: RHS operators
        # ---------------------------------------------------------------------

        # Coriolis force (diagonal)
        cori_op = operator_coriolis_diagonal(op, l, m)
        add_block!(A_rows, A_cols, A_vals, cori_op, row_base, col_base)

        # Viscous diffusion (appears with a minus sign in Kore)
        visc_op = operator_viscous_diffusion(op, l, E)
        add_block!(A_rows, A_cols, A_vals, -visc_op, row_base, col_base)

        # Buoyancy (coupling from temperature)
        buoy_op = operator_buoyancy(op, l, Ra, Pr)
        temp_col_base = (nb_u + nb_v + nb_f + nb_g + k - 1) * n_per_mode
        add_block!(A_rows, A_cols, A_vals, buoy_op, row_base, temp_col_base)

        # Lorentz force from magnetic field (if Le > 0)
        if Le > 0
            # Coupling from poloidal magnetic field (bpol, section f)
            for offset in -2:2
                l_coupled = l + offset
                if l_coupled in op.ll_f
                    k_f = findfirst(==(l_coupled), op.ll_f)
                    f_col_base = (nb_u + nb_v + k_f - 1) * n_per_mode
                    lorentz_bpol = operator_lorentz_poloidal_from_bpol(op, l, m, offset, Le)
                    add_block!(A_rows, A_cols, A_vals, lorentz_bpol, row_base, f_col_base)
                end
            end

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
        add_block!(B_rows, B_cols, B_vals, -v_op, row_base, col_base)

        # ---------------------------------------------------------------------
        # A matrix: RHS operators
        # ---------------------------------------------------------------------

        # Coriolis (diagonal)
        cori_tor = operator_coriolis_toroidal(op, l, m)
        add_block!(A_rows, A_cols, A_vals, cori_tor, row_base, col_base)

        # Viscous diffusion (minus sign following Kore)
        visc_tor = operator_viscous_toroidal(op, l, E)
        add_block!(A_rows, A_cols, A_vals, -visc_tor, row_base, col_base)

        # Lorentz force from magnetic field (if Le > 0)
        if Le > 0
            # Coupling from poloidal magnetic field (section f, offsets l-1:l+1)
            for offset in -1:1
                l_src = l + offset
                idx_f = findfirst(==(l_src), op.ll_f)
                idx_f === nothing && continue
                f_col_base = (nb_u + nb_v + idx_f - 1) * n_per_mode
                lorentz_from_bpol = operator_lorentz_toroidal_from_bpol(op, l, m, offset, Le)
                add_block!(A_rows, A_cols, A_vals, lorentz_from_bpol, row_base, f_col_base)
            end

            # Coupling from toroidal magnetic field (section g, offsets l-2:l+2)
            for offset in -2:2
                l_src = l + offset
                idx_g = findfirst(==(l_src), op.ll_g)
                idx_g === nothing && continue
                g_col_base = (nb_u + nb_v + nb_f + idx_g - 1) * n_per_mode
                lorentz_from_btor = operator_lorentz_toroidal_from_btor(op, l, m, offset, Le)
                add_block!(A_rows, A_cols, A_vals, lorentz_from_btor, row_base, g_col_base)
            end
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
    if nb_f > 0
        println("  Assembling section f (poloidal magnetic field)...")

        for (k, l) in enumerate(op.ll_f)
            row_base = (nb_u + nb_v + k - 1) * n_per_mode
            col_base = (nb_u + nb_v + k - 1) * n_per_mode

            # -----------------------------------------------------------------
            # B matrix: Time derivative
            # -----------------------------------------------------------------
            b_pol = operator_b_poloidal(op, l)
            add_block!(B_rows, B_cols, B_vals, b_pol, row_base, col_base)

            # -----------------------------------------------------------------
            # A matrix: RHS operators
            # -----------------------------------------------------------------

            # Magnetic diffusion
            mag_diff_pol = operator_magnetic_diffusion_poloidal(op, l, Etherm)
            add_block!(A_rows, A_cols, A_vals, mag_diff_pol, row_base, col_base)

            # Induction from velocity field
            if Le > 0
                # From poloidal velocity u (offsets l-2 ... l+2)
                for offset in -2:2
                    l_src = l + offset
                    idx_u = findfirst(==(l_src), op.ll_u)
                    idx_u === nothing && continue

                    induct_from_u = operator_induction_poloidal_from_u(op, l, m, offset)
                    u_col_base = (idx_u - 1) * n_per_mode
                    add_block!(A_rows, A_cols, A_vals, induct_from_u, row_base, u_col_base)
                end

                # From toroidal velocity v (offsets l-1 ... l+1)
                for offset in -1:1
                    l_src = l + offset
                    idx_v = findfirst(==(l_src), op.ll_v)
                    idx_v === nothing && continue

                    induct_from_v = operator_induction_poloidal_from_v(op, l, m, offset)
                    v_col_base = (nb_u + idx_v - 1) * n_per_mode
                    add_block!(A_rows, A_cols, A_vals, induct_from_v, row_base, v_col_base)
                end
            end
        end
    end

    # =========================================================================
    # SECTION G: Toroidal Magnetic Field (1curl induction)
    # =========================================================================
    if nb_g > 0
        println("  Assembling section g (toroidal magnetic field)...")

        for (k, l) in enumerate(op.ll_g)
            row_base = (nb_u + nb_v + nb_f + k - 1) * n_per_mode
            col_base = (nb_u + nb_v + nb_f + k - 1) * n_per_mode

            # -----------------------------------------------------------------
            # B matrix: Time derivative
            # -----------------------------------------------------------------
            b_tor = operator_b_toroidal(op, l)
            add_block!(B_rows, B_cols, B_vals, b_tor, row_base, col_base)

            # -----------------------------------------------------------------
            # A matrix: RHS operators
            # -----------------------------------------------------------------

            # Magnetic diffusion
            mag_diff_tor = operator_magnetic_diffusion_toroidal(op, l, Em)
            add_block!(A_rows, A_cols, A_vals, mag_diff_tor, row_base, col_base)

            # Induction from velocity field (if Le > 0)
            if Le > 0
                # From toroidal velocity v (offsets l-2 ... l+2)
                for offset in -2:2
                    l_src = l + offset
                    idx_v = findfirst(==(l_src), op.ll_v)
                    idx_v === nothing && continue
                    v_col_base = (nb_u + idx_v - 1) * n_per_mode
                    induct_v_tor = operator_induction_toroidal_from_v(op, l, m, offset)
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
    if nb_f > 0
        apply_magnetic_boundary_conditions!(A, B, op, :f)
    end
    if nb_g > 0
        apply_magnetic_boundary_conditions!(A, B, op, :g)
    end

    # Temperature BCs (same as hydrodynamic)
    apply_temperature_boundary_conditions!(A, B, op)

    println("  Final A sparsity: $(nnz(A)) / $(n^2)")
    println("  Final B sparsity: $(nnz(B)) / $(n^2)")

    # Identify interior DOFs
    B_diag = diag(B)
    interior_dofs = findall(i -> abs(B_diag[i]) > 1e-14, 1:n)
    println("  Interior DOFs: $(length(interior_dofs)) / $n")

    section_labels = String[]
    nb_u > 0 && push!(section_labels, "u")
    nb_v > 0 && push!(section_labels, "v")
    nb_f > 0 && push!(section_labels, "f")
    nb_g > 0 && push!(section_labels, "g")
    nb_h > 0 && push!(section_labels, "h")

    info = Dict(
        "method" => "MHD sparse ultraspherical",
        "N" => N,
        "lmax" => params.lmax,
        "m" => m,
        "nl_modes" => op.nl_modes,
        "matrix_size" => n,
        "sections" => join(section_labels, ", ")
    )

    return A, B, interior_dofs, info
end

# -----------------------------------------------------------------------------
# Boundary condition helpers
# -----------------------------------------------------------------------------

function apply_velocity_boundary_conditions!(A, B, op)
    # Apply boundary conditions to velocity fields (poloidal and toroidal)
    # Following the correct implementation from SparseOperator.jl
    params = op.params
    N = params.N
    n_per_mode = N + 1
    nb_u = length(op.ll_u)
    ro = one(params.E)

    # -------------------------------------------------------------------------
    # Poloidal velocity BCs (section u)
    # -------------------------------------------------------------------------
    for (k, l) in enumerate(op.ll_u)
        row_base = (k - 1) * n_per_mode

        # Outer boundary (r = ro = 1.0)
        if params.bco == 1
            # No-slip: u = 0, du/dr = 0
            UltrasphericalSpectral.apply_boundary_conditions!(A, B, [row_base + 1], :dirichlet, N,
                                                              params.ricb, ro)
            UltrasphericalSpectral.apply_boundary_conditions!(A, B, [row_base + 2], :neumann, N,
                                                              params.ricb, ro)
        else
            # Stress-free: u = 0, r·d²u/dr² = 0
            UltrasphericalSpectral.apply_boundary_conditions!(A, B, [row_base + 1], :dirichlet, N,
                                                              params.ricb, ro)
            UltrasphericalSpectral.apply_boundary_conditions!(A, B, [row_base + 2], :neumann2, N,
                                                              params.ricb, ro)
        end

        # Inner boundary (r = ri = ricb)
        if params.bci == 1
            # No-slip: u = 0, du/dr = 0
            UltrasphericalSpectral.apply_boundary_conditions!(A, B, [row_base + n_per_mode], :dirichlet, N,
                                                              params.ricb, ro)
            UltrasphericalSpectral.apply_boundary_conditions!(A, B, [row_base + n_per_mode - 1], :neumann, N,
                                                              params.ricb, ro)
        else
            # Stress-free: u = 0, r·d²u/dr² = 0
            UltrasphericalSpectral.apply_boundary_conditions!(A, B, [row_base + n_per_mode], :dirichlet, N,
                                                              params.ricb, ro)
            UltrasphericalSpectral.apply_boundary_conditions!(A, B, [row_base + n_per_mode - 1], :neumann2, N,
                                                              params.ricb, ro)
        end
    end

    # -------------------------------------------------------------------------
    # Toroidal velocity BCs (section v)
    # -------------------------------------------------------------------------
    scale = UltrasphericalSpectral._radial_scale(params.ricb, ro)
    outer_vals = UltrasphericalSpectral._chebyshev_boundary_values(N, :outer)
    inner_vals = UltrasphericalSpectral._chebyshev_boundary_values(N, :inner)
    outer_deriv = UltrasphericalSpectral._chebyshev_boundary_derivative(N, :outer)
    inner_deriv = UltrasphericalSpectral._chebyshev_boundary_derivative(N, :inner)
    r_outer = UltrasphericalSpectral._boundary_radius(params.ricb, ro, :outer)
    r_inner = UltrasphericalSpectral._boundary_radius(params.ricb, ro, :inner)
    outer_row = @. -r_outer * scale * outer_deriv + outer_vals
    inner_row = @. -r_inner * scale * inner_deriv + inner_vals

    for (k, l) in enumerate(op.ll_v)
        row_base = (nb_u + k - 1) * n_per_mode

        # Outer boundary (r = ro = 1.0)
        if params.bco == 1
            # No-slip: v = 0
            UltrasphericalSpectral.apply_boundary_conditions!(A, B, [row_base + 1], :dirichlet, N,
                                                              params.ricb, ro)
        else
            # Stress-free: -r·∂v/∂r + v = 0
            row = row_base + 1
            A[row, :] .= 0.0
            B[row, :] .= 0.0
            block_start = row_base + 1
            A[row, block_start:(block_start + N)] = outer_row
        end

        # Inner boundary (r = ri = ricb)
        if params.bci == 1
            # No-slip: v = 0
            UltrasphericalSpectral.apply_boundary_conditions!(A, B, [row_base + n_per_mode], :dirichlet, N,
                                                              params.ricb, ro)
        else
            # Stress-free: -r·∂v/∂r + v = 0
            row = row_base + n_per_mode
            A[row, :] .= 0.0
            B[row, :] .= 0.0
            block_start = row_base + 1
            A[row, block_start:(block_start + N)] = inner_row
        end
    end
end

function apply_temperature_boundary_conditions!(A, B, op)
    # Apply boundary conditions to temperature field
    # Following the correct implementation from SparseOperator.jl
    params = op.params
    N = params.N
    n_per_mode = N + 1
    nb_u = length(op.ll_u)
    nb_v = length(op.ll_v)
    nb_f = length(op.ll_f)
    nb_g = length(op.ll_g)
    ro = one(params.E)

    # -------------------------------------------------------------------------
    # Temperature BCs (section h)
    # -------------------------------------------------------------------------
    for (k, l) in enumerate(op.ll_h)
        row_base = (nb_u + nb_v + nb_f + nb_g + k - 1) * n_per_mode

        # Outer boundary (r = ro = 1.0)
        if params.bco_thermal == 0
            # Fixed temperature: θ = 0
            UltrasphericalSpectral.apply_boundary_conditions!(A, B, [row_base + 1], :dirichlet, N,
                                                              params.ricb, ro)
        else
            # Fixed flux: dθ/dr = 0
            UltrasphericalSpectral.apply_boundary_conditions!(A, B, [row_base + 1], :neumann, N,
                                                              params.ricb, ro)
        end

        # Inner boundary (r = ri = ricb)
        if params.bci_thermal == 0
            # Fixed temperature: θ = 0
            UltrasphericalSpectral.apply_boundary_conditions!(A, B, [row_base + n_per_mode], :dirichlet, N,
                                                              params.ricb, ro)
        else
            # Fixed flux: dθ/dr = 0
            UltrasphericalSpectral.apply_boundary_conditions!(A, B, [row_base + n_per_mode], :neumann, N,
                                                              params.ricb, ro)
        end
    end
end
