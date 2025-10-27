#!/usr/bin/env julia
using LinearAlgebra
using SparseArrays

const ROOT = normpath(joinpath(@__DIR__, ".."))
ENV["PYTHON"] = joinpath(ROOT, ".venv", "bin", "python")

using PyCall
push!(PyVector(pyimport("sys")["path"]), @__DIR__)
kf = pyimport("kore_formulas")
np = pyimport("numpy")

include(joinpath(ROOT, "src", "SparseOperator.jl"))
const SO = SparseOperator

function np_array(mat::SparseMatrixCSC)
    return np.array(Matrix(mat))
end

function maxabsdiff(A::AbstractArray, B::AbstractArray)
    return maximum(abs.(A .- B))
end

function compare_params(params)
    op = SO.SparseStabilityOperator(params)
    println("Comparing for bci=$(params.bci), bco=$(params.bco), heating=$(params.heating)")

    # Poloidal modes
    for l in op.ll_top
        L = l * (l + 1)
        r2_D0_u = np_array(op.r2_D0_u)
        r3_D1_u = np_array(op.r3_D1_u)
        r4_D2_u = np_array(op.r4_D2_u)
        r0_D0_u = np_array(op.r0_D0_u)
        r2_D2_u = np_array(op.r2_D2_u)
        r3_D3_u = np_array(op.r3_D3_u)
        r4_D4_u = np_array(op.r4_D4_u)

        u_py = Array(kf.operator_u(l, r2_D0_u, r3_D1_u, r4_D2_u))
        u_jl = Matrix(SO.operator_u(op, l))
        println("  l=$(l) operator_u diff: ", maxabsdiff(u_py, u_jl))

        cori_py = Array(kf.operator_coriolis_diagonal(l, params.m, r2_D0_u, r3_D1_u, r4_D2_u))
        cori_jl = Matrix(SO.operator_coriolis_diagonal(op, l, params.m))
        println("    coriolis diff: ", maxabsdiff(cori_py, cori_jl))

        visc_py = Array(kf.operator_viscous_diffusion(l, params.E, r0_D0_u, r2_D2_u, r3_D3_u, r4_D4_u))
        visc_jl = Matrix(SO.operator_viscous_diffusion(op, l, params.E))
        println("    viscous diff: ", maxabsdiff(visc_py, visc_jl))

        for offset in (-1, 1)
            r3_D0_u = np_array(op.r3_D0_u)
            r4_D1_u = np_array(op.r4_D1_u)
            py_mat = Array(kf.operator_coriolis_offdiag(l, params.m, offset, r3_D0_u, r4_D1_u))
            jl_mat, _ = SO.operator_coriolis_offdiag(op, l, params.m, offset)
            println("    coriolis offdiag offset=$(offset) diff: ", maxabsdiff(py_mat, Matrix(jl_mat)))
        end

        buoy_py = Array(kf.operator_buoyancy(l, params.Ra, params.E, params.Pr, np_array(op.r4_D0_u)))
        buoy_jl = Matrix(SO.operator_buoyancy(op, l, params.Ra, params.Pr))
        println("    buoy diff: ", maxabsdiff(buoy_py, buoy_jl))
    end

    # Toroidal modes
    for l in op.ll_bot
        r2_D0_v = np_array(op.r2_D0_v)
        r0_D0_v = np_array(op.r0_D0_v)
        r1_D1_v = np_array(op.r1_D1_v)
        r2_D2_v = np_array(op.r2_D2_v)
        r1_D0_v = np_array(op.r1_D0_v)
        r2_D1_v = np_array(op.r2_D1_v)

        u_tor_py = Array(kf.operator_u_toroidal(l, r2_D0_v))
        u_tor_jl = Matrix(SO.operator_u_toroidal(op, l))
        println("  l=$(l) operator_u_tor diff: ", maxabsdiff(u_tor_py, u_tor_jl))

        cori_tor_py = Array(kf.operator_coriolis_toroidal(params.m, r2_D0_v))
        cori_tor_jl = Matrix(SO.operator_coriolis_toroidal(op, l, params.m))
        println("    coriolis diff: ", maxabsdiff(cori_tor_py, cori_tor_jl))

        visc_tor_py = Array(kf.operator_viscous_toroidal(l, params.E, r0_D0_v, r1_D1_v, r2_D2_v))
        visc_tor_jl = Matrix(SO.operator_viscous_toroidal(op, l, params.E))
        println("    viscous diff: ", maxabsdiff(visc_tor_py, visc_tor_jl))

        for offset in (-1, 1)
            py_mat = Array(kf.operator_coriolis_v_to_u(l, params.m, offset, r1_D0_v, r2_D1_v))
            jl_mat = SO.operator_coriolis_v_to_u(op, l, params.m, offset)
            println("    coriolis v→u offset=$(offset) diff: ", maxabsdiff(py_mat, Matrix(jl_mat)))
        end
    end

    # Temperature
    heating_str = string(params.heating)
    r2_D0_h = np_array(op.r2_D0_h)
    r3_D0_h = np_array(op.r3_D0_h)
    r0_D0_h = np_array(op.r0_D0_h)
    r1_D0_h = np_array(op.r1_D0_h)
    r1_D1_h = np_array(op.r1_D1_h)
    r2_D1_h = np_array(op.r2_D1_h)
    r2_D2_h = np_array(op.r2_D2_h)
    r3_D2_h = np_array(op.r3_D2_h)

    for l in op.ll_top
        theta_py = Array(kf.operator_theta(heating_str, r2_D0_h, r3_D0_h))
        theta_jl = Matrix(SO.operator_theta(op, l))
        println("  Temp l=$(l) theta diff: ", maxabsdiff(theta_py, theta_jl))

        diff_py = Array(kf.operator_thermal_diffusion(l, params.Etherm, heating_str, r0_D0_h, r1_D0_h, r1_D1_h, r2_D1_h, r2_D2_h, r3_D2_h))
        diff_jl = Matrix(SO.operator_thermal_diffusion(op, l, params.Etherm))
        println("    thermal diffusion diff: ", maxabsdiff(diff_py, diff_jl))

        adv_py = Array(kf.operator_thermal_advection(l, heating_str, params.ricb, r0_D0_h, r2_D0_h))
        adv_jl = Matrix(SO.operator_thermal_advection(op, l))
        println("    thermal advection diff: ", maxabsdiff(adv_py, adv_jl))
    end
end

params_list = [
    SO.SparseOnsetParams(E=1e-4, Pr=1.0, Ra=1e5, ricb=0.35, m=2, lmax=40, N=40,
                         bci=1, bco=1, bci_thermal=0, bco_thermal=0),
    SO.SparseOnsetParams(E=1e-4, Pr=1.0, Ra=1e5, ricb=0.35, m=2, lmax=40, N=40,
                         bci=0, bco=0, bci_thermal=0, bco_thermal=0,
                         heating=:internal)
]

for params in params_list
    compare_params(params)
end
