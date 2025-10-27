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
        L = l * (l + 1)
        # B matrix
        block = L * (L * op.r2_D0_u - 2 * op.r3_D1_u - op.r4_D2_u)
        add_block!(B, block, row_base, col_base)
        # Coriolis
        block = 2im * m * (-L * op.r2_D0_u + 2 * op.r3_D1_u + op.r4_D2_u)
        add_block!(A, block, row_base, col_base)
        # Viscous
        block = E * L * (-L * op.r0_D0_u + 2 * L * op.r2_D2_u - 4 * op.r3_D3_u - op.r4_D4_u)
        add_block!(A, block, row_base, col_base)
        # Off-diagonal Coriolis
        for offset in (-1, 1)
            l2 = l + offset
            idx = findfirst(==(l2), op.ll_bot)
            idx === nothing && continue
            col_off = (nb_top + idx - 1) * n_per_mode
            if offset == -1
                C = (l^2 - 1) * sqrt(l^2 - m^2) / (2l - 1)
                block = 2 * C * ((l - 1) * op.r3_D0_u - op.r4_D1_u)
            else
                C = l * (l + 2) * sqrt((l + m + 1) * (l - m + 1)) / (2l + 3)
                block = 2 * C * (-(l + 2) * op.r3_D0_u - op.r4_D1_u)
            end
            add_block!(A, block, row_base, col_off)
        end
        # Buoyancy
        beyonce = -Ra * E^2 / Pr
        block = beyonce * L * op.r4_D0_u
        temp_col = (nb_top + nb_bot + k - 1) * n_per_mode
        add_block!(A, block, row_base, temp_col)
    end

    for (k, l) in enumerate(op.ll_bot)
        row_base = (nb_top + k - 1) * n_per_mode
        col_base = (nb_top + k - 1) * n_per_mode
        L = l * (l + 1)
        # B matrix
        block = L * op.r2_D0_v
        add_block!(B, block, row_base, col_base)
        # Coriolis diagonal
        block = -2im * m * op.r2_D0_v
        add_block!(A, block, row_base, col_base)
        # Viscous
        block = E * L * (-L * op.r0_D0_v + 2 * op.r1_D1_v + op.r2_D2_v)
        add_block!(A, block, row_base, col_base)
        # Off-diagonal v->u
        for offset in (-1, 1)
            l2 = l + offset
            idx = findfirst(==(l2), op.ll_top)
            idx === nothing && continue
            col_off = (idx - 1) * n_per_mode
            if offset == -1
                C = (l^2 - 1) * sqrt(l^2 - m^2) / (2l - 1)
                block = 2 * C * ((l - 1) * op.r1_D0_v - op.r2_D1_v)
            else
                C = l * (l + 2) * sqrt((l + m + 1) * (l - m + 1)) / (2l + 3)
                block = 2 * C * (-(l + 2) * op.r1_D0_v - op.r2_D1_v)
            end
            add_block!(A, block, row_base, col_off)
        end
    end

    for (k, l) in enumerate(op.ll_top)
        row_base = (nb_top + nb_bot + k - 1) * n_per_mode
        col_base = (nb_top + nb_bot + k - 1) * n_per_mode
        block = op.r2_D0_h
        if op.params.heating == :differential
            block = op.r3_D0_h
        end
        add_block!(B, block, row_base, col_base)
        L = l * (l + 1)
        if op.params.heating == :differential
            block = params.Etherm * (-L * op.r1_D0_h + 2 * op.r2_D1_h + op.r3_D2_h)
        else
            block = params.Etherm * (-L * op.r0_D0_h + 2 * op.r1_D1_h + op.r2_D2_h)
        end
        add_block!(A, block, row_base, col_base)
        block = L * op.r2_D0_h
        if op.params.heating == :differential
            block = L * (op.params.ricb / (one(eltype(block)) - op.params.ricb)) * op.r0_D0_h
        end
        vel_col = (k - 1) * n_per_mode
        add_block!(A, block, row_base, vel_col)
    end

    return A, B
end

function compare_with_reference(params)
    op = SO.SparseStabilityOperator(params)
    A1, B1, _, _ = SO.assemble_sparse_matrices(op)
    A2, B2 = assemble_reference(op)
    # Apply BCs to reference matrices
    SO.apply_sparse_boundary_conditions!(A2, B2, op)
    maxA = maximum(abs.(A1 .- A2))
    maxB = maximum(abs.(B1 .- B2))
    println("max |A_cross - A_ref| = ", maxA)
    println("max |B_cross - B_ref| = ", maxB)
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
