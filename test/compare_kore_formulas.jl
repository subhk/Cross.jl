#!/usr/bin/env julia
using LinearAlgebra
using SparseArrays
include("../src/SparseOperator.jl")
const SO = SparseOperator

function assemble_reference(op::SO.SparseStabilityOperator)
    params = op.params
    N = params.N
    m = params.m
    E = params.E
    Pr = params.Pr
    Ra = params.Ra

    n_per_mode = N + 1
    nb_top = length(op.ll_top)
    nb_bot = length(op.ll_bot)
    n = op.matrix_size

    A = spzeros(ComplexF64, n, n)
    B = spzeros(ComplexF64, n, n)

    function add_block!(M::SparseMatrixCSC, block::SparseMatrixCSC, r0::Int, c0::Int)
        I, J, V = findnz(block)
        for k in eachindex(V)
            M[r0 + I[k], c0 + J[k]] += ComplexF64(V[k])
        end
    end

    for (k, l) in enumerate(op.ll_top)
        row_base = (k - 1) * n_per_mode
        col_base = (k - 1) * n_per_mode
        add_block!(B, -SO.operator_u(op, l), row_base, col_base)
        add_block!(A, SO.operator_coriolis_diagonal(op, l, m), row_base, col_base)
        add_block!(A, -SO.operator_viscous_diffusion(op, l, E), row_base, col_base)
        # Off-diagonal Coriolis
        for offset in (-1, 1)
            l2 = l + offset
            idx = findfirst(==(l2), op.ll_bot)
            idx === nothing && continue
            col_off = (nb_top + idx - 1) * n_per_mode
            block, _ = SO.operator_coriolis_offdiag(op, l, m, offset)
            add_block!(A, block, row_base, col_off)
        end
        block = SO.operator_buoyancy(op, l, Ra, Pr)
        temp_col = (nb_top + nb_bot + k - 1) * n_per_mode
        add_block!(A, block, row_base, temp_col)
    end

    for (k, l) in enumerate(op.ll_bot)
        row_base = (nb_top + k - 1) * n_per_mode
        col_base = (nb_top + k - 1) * n_per_mode
        add_block!(B, -SO.operator_u_toroidal(op, l), row_base, col_base)
        add_block!(A, SO.operator_coriolis_toroidal(op, l, m), row_base, col_base)
        add_block!(A, -SO.operator_viscous_toroidal(op, l, E), row_base, col_base)
        # Off-diagonal v->u
        for offset in (-1, 1)
            l2 = l + offset
            idx = findfirst(==(l2), op.ll_top)
            idx === nothing && continue
            col_off = (idx - 1) * n_per_mode
            block = SO.operator_coriolis_v_to_u(op, l, m, offset)
            add_block!(A, block, row_base, col_off)
        end
    end

    for (k, l) in enumerate(op.ll_top)
        row_base = (nb_top + nb_bot + k - 1) * n_per_mode
        col_base = (nb_top + nb_bot + k - 1) * n_per_mode
        add_block!(B, SO.operator_theta(op, l), row_base, col_base)
        add_block!(A, SO.operator_thermal_diffusion(op, l, params.Etherm), row_base, col_base)
        vel_col = (k - 1) * n_per_mode
        add_block!(A, SO.operator_thermal_advection(op, l), row_base, vel_col)
    end

    return A, B
end

function compare_with_reference(params)
    op = SO.SparseStabilityOperator(params)
    A1, B1, _, _ = SO.assemble_sparse_matrices(op)
    A2, B2 = assemble_reference(op)
    # Apply BCs to reference matrices
    SO.apply_sparse_boundary_conditions!(A2, B2, op)
    diffA = A1 - A2
    diffB = B1 - B2
    maxA, idxA = findmax(abs.(diffA))
    maxB, idxB = findmax(abs.(diffB))
    println("max |A_cross - A_ref| = ", maxA)
    println("  location: (", idxA[1], ", ", idxA[2], ") value = ", diffA[idxA])
    println("max |B_cross - B_ref| = ", maxB)
    println("  location: (", idxB[1], ", ", idxB[2], ") value = ", diffB[idxB])
end

params_list = [
    SO.SparseOnsetParams(E=1e-4, Pr=1.0, Ra=1e5, ricb=0.35, m=2, lmax=6, N=8,
                         bci=1, bco=1, bci_thermal=0, bco_thermal=0),
    SO.SparseOnsetParams(E=1e-4, Pr=1.0, Ra=1e5, ricb=0.35, m=2, lmax=6, N=8,
                         bci=0, bco=0, bci_thermal=0, bco_thermal=0)
]

for params in params_list
    println("Testing parameters: bci=$(params.bci), bco=$(params.bco)")
    compare_with_reference(params)
end
