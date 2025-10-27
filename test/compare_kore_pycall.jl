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
include(joinpath(ROOT, "src", "CompleteMHD.jl"))
const CM = CompleteMHD

function np_array(mat::SparseMatrixCSC)
    return np.array(Matrix(mat))
end

function add_matrix!(dict::PyDict, name::String, mat)
    dict[Symbol(name)] = np.array(Matrix{ComplexF64}(mat))
end

function build_mhd_matrix_dict(op::CM.MHDStabilityOperator)
    mats = PyDict()

    # Base radial operators for magnetic sections
    add_matrix!(mats, "r0_D0_f", op.r0_D0_f)
    add_matrix!(mats, "r1_D1_f", op.r1_D1_f)
    add_matrix!(mats, "r2_D2_f", op.r2_D2_f)
    add_matrix!(mats, "r2_D0_f", op.r2_D0_f)
    add_matrix!(mats, "r0_D0_g", op.r0_D0_g)
    add_matrix!(mats, "r1_D1_g", op.r1_D1_g)
    add_matrix!(mats, "r2_D2_g", op.r2_D2_g)
    add_matrix!(mats, "r2_D0_g", op.r2_D0_g)

    # Background operators for velocity (u/v)
    for p in -1:5, h in 0:3, d in 0:3
        name_u = "r$(p)_h$(h)_D$(d)_u"
        name_v = "r$(p)_h$(h)_D$(d)_v"
        mat = CM.background_operator(op, p, h, d)
        add_matrix!(mats, name_u, mat)
        add_matrix!(mats, name_v, mat)
    end

    # Background operators for magnetic poloidal/toroidal sections
    for p in -1:5, h in 0:2, d in 0:2
        add_matrix!(mats, "r$(p)_h$(h)_D$(d)_f", CM.background_operator(op, p, h, d))
        add_matrix!(mats, "r$(p)_h$(h)_D$(d)_g", CM.background_operator(op, p, h, d))
    end

    return mats
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

function compare_mhd(params)
    op = CM.MHDStabilityOperator(params)
    println("Comparing MHD operators for Le=$(params.Le), B0=$(params.B0_type)")
    mats = build_mhd_matrix_dict(op)
    m = params.m
    Le2 = params.Le^2
    Em = params.E / params.Pm

    for l in op.ll_u
        mat_diag_py = Array(kf.lorentz_upol_diag_axial(l, m, mats))
        mat_diag_jl = Matrix(CM.operator_lorentz_poloidal_diagonal(op, l, params.Le))
        println("  l=$(l) Lorentz diag diff: ", maxabsdiff(Le2 * mat_diag_py, mat_diag_jl))

        for offset in -2:2
            mat_py = Array(kf.lorentz_upol_bpol_axial(l, m, offset, mats))
            mat_jl = Matrix(CM.operator_lorentz_poloidal_from_bpol(op, l, m, offset, params.Le))
            println("    offset=$(offset) bpol diff: ", maxabsdiff(Le2 * mat_py, mat_jl))
        end

        for offset in (-1, 1)
            mat_py = Array(kf.lorentz_upol_btor_axial(l, m, offset, mats))
            mat_jl = Matrix(CM.operator_lorentz_poloidal_offdiag(op, l, m, offset, params.Le))
            println("    offset=$(offset) btor diff: ", maxabsdiff(Le2 * mat_py, mat_jl))
        end

        for offset in -2:2
            mat_py = Array(kf.induction_f_upol_axial(l, m, offset, mats))
            mat_jl = Matrix(CM.operator_induction_poloidal_from_u(op, l, m, offset))
            println("    induction (u) offset=$(offset) diff: ", maxabsdiff(mat_py, mat_jl))
        end

        for offset in -1:1
            mat_py = Array(kf.induction_f_utor_axial(l, m, offset, mats))
            mat_jl = Matrix(CM.operator_induction_poloidal_from_v(op, l, m, offset))
            println("    induction (v) offset=$(offset) diff: ", maxabsdiff(mat_py, mat_jl))
        end

        diff_py = Array(kf.magnetic_diffusion_f_axial(l, params.Etherm, mats))
        diff_jl = Matrix(CM.operator_magnetic_diffusion_poloidal(op, l, Em))
        println("    magnetic diffusion (f) diff: ", maxabsdiff(diff_py, diff_jl))

        b_py = Array(kf.b_poloidal_axial(l, mats))
        b_jl = Matrix(CM.operator_b_poloidal(op, l))
        println("    b_poloidal diff: ", maxabsdiff(b_py, b_jl))
    end

    for l in op.ll_v
        lorentz_py = Array(kf.lorentz_utor_axial(l, m, mats))
        lorentz_jl = Matrix(CM.operator_lorentz_toroidal(op, l, params.Le))
        println("  l=$(l) Lorentz tor diff: ", maxabsdiff(Le2 * lorentz_py, lorentz_jl))

        for offset in -1:1
            mat_py = Array(kf.lorentz_v_bpol_axial(l, m, offset, mats))
            mat_jl = Matrix(CM.operator_lorentz_toroidal_from_bpol(op, l, m, offset, params.Le))
            println("    lorentz v←bpol offset=$(offset) diff: ", maxabsdiff(Le2 * mat_py, mat_jl))
        end

        for offset in -2:2
            mat_py = Array(kf.lorentz_v_btor_axial(l, m, offset, mats))
            mat_jl = Matrix(CM.operator_lorentz_toroidal_from_btor(op, l, m, offset, params.Le))
            println("    lorentz v←btor offset=$(offset) diff: ", maxabsdiff(Le2 * mat_py, mat_jl))
        end

        for offset in (-1, 1)
            mat_py = Array(kf.induction_g_upol_axial(l, m, offset, mats))
            mat_jl = Matrix(CM.operator_induction_toroidal_from_u(op, l, m, offset))
            println("    induction tor<-u offset=$(offset) diff: ", maxabsdiff(mat_py, mat_jl))
        end

        for offset in -2:2
            mat_py = Array(kf.induction_g_utor_axial(l, m, offset, mats))
            mat_jl = Matrix(CM.operator_induction_toroidal_from_v(op, l, m, offset))
            println("    induction tor<-v offset=$(offset) diff: ", maxabsdiff(mat_py, mat_jl))
        end

        diff_py = Array(kf.magnetic_diffusion_g_axial(l, Em, mats))
        diff_jl = Matrix(CM.operator_magnetic_diffusion_toroidal(op, l, Em))
        println("    magnetic diffusion (g) diff: ", maxabsdiff(diff_py, diff_jl))

        b_py = Array(kf.b_toroidal_axial(l, mats))
        b_jl = Matrix(CM.operator_b_toroidal(op, l))
        println("    b_toroidal diff: ", maxabsdiff(b_py, b_jl))
    end
end

params_mhd = CM.MHDParams(
    E = 1e-3,
    Pr = 1.0,
    Pm = 5.0,
    Ra = 1e4,
    Le = 0.1,
    ricb = 0.35,
    m = 2,
    lmax = 6,
    N = 8,
    B0_type = CM.axial,
    bci = 1,
    bco = 1,
    bci_magnetic = 0,
    bco_magnetic = 0,
    heating = :differential
)

compare_mhd(params_mhd)
