#!/usr/bin/env julia
#
# Cross-check MHD operator blocks against Kore's analytical formulas
# (restricted to axial background field, B0 = axial, anelastic = 0).
#
# This mirrors the structure of kore-main/bin/operators.py for the magnetic
# terms and verifies that the linear combinations of background operators match
# exactly.

using LinearAlgebra
using SparseArrays
using Test

include("../src/CompleteMHD.jl")
using .CompleteMHD

const TOL = 1e-12

function dense(mat)
    Matrix{ComplexF64}(mat)
end

function normdiff(A::AbstractArray, B::AbstractArray)
    maximum(abs.(A .- B))
end

function setup_operator()
    params = MHDParams(
        E = 1e-3,
        Pr = 1.0,
        Pm = 5.0,
        Ra = 1e4,
        Le = 0.1,
        ricb = 0.35,
        m = 2,
        lmax = 6,
        N = 8,
        B0_type = axial,
        bci = 1,
        bco = 1,
        bci_magnetic = 0,
        bco_magnetic = 0,
        heating = :differential
    )
    return MHDStabilityOperator(params)
end

function bo_u(op, p, h, d)
    dense(CompleteMHD.background_operator(op, p, h, d))
end

const ZERO = ComplexF64(0)

function lorentz_upol_expected(op, l, m, offset)
    L = l * (l + 1)
    Le2 = op.params.Le^2

    if offset == 0
        C = 3 * (l + l^2 - 3m^2) / (-3 + 4l * (1 + l))
        combo = C * (
            3 * L * (1 + l) * (-2 + l + l^2) .* bo_u(op, 0, 0, 0) .+
            (-3 * L^2) .* bo_u(op, 1, 1, 0) .+
            2 * (6 - 4l - 5l^2 - 2l^3 - l^4) .* bo_u(op, 1, 0, 1) .+
            3 * L .* bo_u(op, 2, 2, 0) .+
            (-12 + 5l + 5l^2) .* bo_u(op, 2, 0, 2) .+
            2 * (-6 + 5l + 5l^2) .* bo_u(op, 2, 1, 1) .+
            2 * L .* bo_u(op, 3, 2, 1) .+
            L .* bo_u(op, 3, 3, 0) .+
            2 * (-3 + l + l^2) .* bo_u(op, 3, 0, 3) .+
            3 * (-2 + l + l^2) .* bo_u(op, 3, 1, 2)
        )
        return Le2 .* combo
    elseif offset == -1
        denom = 2l - 1
        C = sqrt(max(l^2 - m^2, 0)) * (l^2 - 1) / denom
        combo = -2 * (l^2 + 2) .* bo_u(op, 1, 0, 1)
        combo .+= -2 * (l - 2) .* bo_u(op, 2, 1, 1)
        combo .+= -(l - 4) .* bo_u(op, 2, 0, 2)
        combo .+= -(l - 2) .* bo_u(op, 3, 1, 2)
        combo .+= L * (l + 2) .* bo_u(op, 0, 0, 0)
        combo .+= L * (l - 4) .* bo_u(op, 1, 1, 0)
        combo .+= l .* bo_u(op, 2, 2, 0)
        combo .+= l .* bo_u(op, 3, 3, 0)
        combo .+= 2 .* bo_u(op, 3, 0, 3)
        return Le2 .* C .* combo
    elseif offset == 1
        denom = 2l + 3
        C = sqrt(max((l + m + 1) * (l - m + 1), 0)) * l * (l + 2) / denom

        combo = -2 * (l^2 + 2l + 3) .* bo_u(op, 1, 0, 1)
        combo .+= 2 * (l + 3) .* bo_u(op, 2, 1, 1)
        combo .+= (l + 5) .* bo_u(op, 2, 0, 2)
        combo .+= (l + 3) .* bo_u(op, 3, 1, 2)
        combo .+= -L * (l - 1) .* bo_u(op, 0, 0, 0)
        combo .+= -L * (l + 5) .* bo_u(op, 1, 1, 0)
        combo .+= -(l + 1) .* bo_u(op, 2, 2, 0)
        combo .+= -(l + 1) .* bo_u(op, 3, 3, 0)
        combo .+= 2 .* bo_u(op, 3, 0, 3)
        return Le2 .* C .* combo
    elseif offset == -2
        denom = 3 - 8l + 4l^2
        C = (3 * (-2 - l + l^2) * sqrt(max((l - m) * (-1 + l + m) * (-1 + l - m) * (l + m), 0))) / denom
        combo = (2l + 3l^2 + l^3) .* bo_u(op, 0, 0, 0)
        combo .+= -(6 - 7l + 3l^2) .* bo_u(op, 1, 0, 1)
        combo .+= (2 + l - 6l^2 + l^3) .* bo_u(op, 1, 1, 0)
        combo .+= (6 - l) .* bo_u(op, 2, 0, 2)
        combo .+= -2 * (-2 + l) .* bo_u(op, 2, 1, 1)
        combo .+= (-2 + l) .* bo_u(op, 2, 2, 0)
        combo .+= -1 .* bo_u(op, 3, 2, 1)
        combo .+= 3 .* bo_u(op, 3, 0, 3)
        combo .+= -( -3 + l) .* bo_u(op, 3, 1, 2)
        combo .+= (-1 + l) .* bo_u(op, 3, 3, 0)
        return Le2 .* C .* combo
    elseif offset == 2
        denom = 15 + 16l + 4l^2
        C = (3 * l * (3 + l) * sqrt(max((1 + l - m) * (2 + l + m) * (1 + l - m) * (2 + l + m), 0))) / denom
        combo = (l - l^3) .* bo_u(op, 0, 0, 0)
        combo .+= -(16 + 13l + 3l^2) .* bo_u(op, 1, 0, 1)
        combo .+= -(6 + 16l + 9l^2 + l^3) .* bo_u(op, 1, 1, 0)
        combo .+= 2 * (3 + l) .* bo_u(op, 2, 1, 1)
        combo .+= -(3 + l) .* bo_u(op, 2, 2, 0)
        combo .+= (7 + l) .* bo_u(op, 2, 0, 2)
        combo .+= -1 .* bo_u(op, 3, 2, 1)
        combo .+= 3 .* bo_u(op, 3, 0, 3)
        combo .+= (4 + l) .* bo_u(op, 3, 1, 2)
        combo .+= -(2 + l) .* bo_u(op, 3, 3, 0)
        return Le2 .* C .* combo
    else
        return zeros(ComplexF64, size(bo_u(op, 0, 0, 0)))
    end
end

function lorentz_utor_expected(op, l, m)
    Le2 = op.params.Le^2
    L = l * (l + 1)
    combo = 4 .* bo_u(op, 0, 0, 1)
    combo .+= -L * (2 .* bo_u(op, 0, 1, 0) .+ bo_u(op, 1, 2, 0))
    combo .+= 2 .* bo_u(op, 1, 0, 2)
    return Le2 * (1im * m) .* combo
end

function lorentz_vbpol_expected(op, l, m, offset)
    Le2 = op.params.Le^2
    if offset == -1
        denom = 2l - 1
        C = (3im * m * sqrt(max(l^2 - m^2, 0))) / denom
        combo = 12 .* bo_u(op, 0, 0, 1)
        combo .+= -2 * (-1 + l) * l .* bo_u(op, 0, 1, 0)
        combo .+= 6 .* bo_u(op, 1, 0, 2)
        combo .+= -(-1 + l) * l .* bo_u(op, 1, 2, 0)
        return Le2 .* C .* combo
    elseif offset == 0
        combo = 4 .* bo_u(op, 0, 0, 1)
        combo .+= -(l * (l + 1) * (2 .* bo_u(op, 0, 1, 0) .+ bo_u(op, 1, 2, 0)))
        combo .+= 2 .* bo_u(op, 1, 0, 2)
        return Le2 * (1im * m) .* combo
    elseif offset == 1
        denom = 2l + 3
        C = (3im * m * sqrt(max((1 + l - m) * (1 + l + m), 0))) / denom
        combo = 12 .* bo_u(op, 0, 0, 1)
        combo .+= -2 * (1 + l) * (2 + l) .* bo_u(op, 0, 1, 0)
        combo .+= 6 .* bo_u(op, 1, 0, 2)
        combo .+= -(1 + l) * (2 + l) .* bo_u(op, 1, 2, 0)
        return Le2 .* C .* combo
    else
        return zeros(ComplexF64, size(bo_u(op, 0, 0, 0)))
    end
end

function lorentz_vbtor_expected(op, l, m, offset)
    Le2 = op.params.Le^2
    L = l * (l + 1)
    if offset == -2
        denom = 3 - 8l + 4l^2
        C = (3 * (l - 2) * (l + 1) * sqrt(max((l - m) * (-1 + l + m) * (-1 + l - m) * (l + m), 0))) / denom
        combo = (-4 + l) .* bo_u(op, 0, 0, 0)
        combo .+= -3 .* bo_u(op, 1, 0, 1)
        combo .+= (-1 + l) .* bo_u(op, 1, 1, 0)
        return Le2 .* C .* combo
    elseif offset == -1
        denom = 2l - 1
        C = sqrt(max((l - m) * (l + m), 0)) * (l^2 - 1) / denom
        combo = (l - 2) .* bo_u(op, 0, 0, 0)
        combo .+= l .* bo_u(op, 1, 1, 0)
        combo .+= -2 .* bo_u(op, 1, 0, 1)
        return Le2 .* C .* combo
    elseif offset == 0
        C = 3 * (l + l^2 - 3m^2) / (-3 + 4l * (1 + l))
        combo = (6 - l - l^2) .* bo_u(op, 0, 0, 0)
        combo .+= L .* bo_u(op, 1, 1, 0)
        combo .+= -2 * (-3 + l + l^2) .* bo_u(op, 1, 0, 1)
        return Le2 .* C .* combo
    elseif offset == 1
        denom = 2l + 3
        C = -sqrt(max((l + m + 1) * (l + 1 - m), 0)) * l * (l + 2) / denom
        combo = (l + 3) .* bo_u(op, 0, 0, 0)
        combo .+= (l + 1) .* bo_u(op, 1, 1, 0)
        combo .+= 2 .* bo_u(op, 1, 0, 1)
        return Le2 .* C .* combo
    elseif offset == 2
        denom = (3 + 2l) * (5 + 2l)
        C = (3 * l * (3 + l) * sqrt(max((2 + l - m) * (1 + l + m) * (1 + l - m) * (2 + l + m), 0))) / denom
        combo = -(5 + l) .* bo_u(op, 0, 0, 0)
        combo .+= -3 .* bo_u(op, 1, 0, 1)
        combo .+= -(2 + l) .* bo_u(op, 1, 1, 0)
        return Le2 .* C .* combo
    else
        return zeros(ComplexF64, size(bo_u(op, 0, 0, 0)))
    end
end

function induction_f_upol_expected(op, l, m, offset)
    if offset == -2
        denom = 3 - 8l + 4l^2
        C = 3 * (l - 2) * (l + 1) * sqrt(max((l - m) * (-1 + l + m) * (-1 + l - m) * (l + m), 0)) / denom
        combo = (-4 + l) .* bo_u(op, 0, 0, 0)
        combo .+= -3 .* bo_u(op, 1, 0, 1)
        combo .+= (-1 + l) .* bo_u(op, 1, 1, 0)
        return combo * C
    elseif offset == -1
        denom = 2l - 1
        C = sqrt(max(l^2 - m^2, 0)) * (l^2 - 1) / denom
        combo = (l - 2) .* bo_u(op, 0, 0, 0)
        combo .+= l .* bo_u(op, 1, 1, 0)
        combo .+= -2 .* bo_u(op, 1, 0, 1)
        return combo * C
    elseif offset == 0
        C = 3 * (l + l^2 - 3m^2) / (-3 + 4l * (1 + l))
        combo = (6 - l - l^2) .* bo_u(op, 0, 0, 0)
        combo .+= l * (1 + l) .* bo_u(op, 1, 1, 0)
        combo .+= -2 * (-3 + l + l^2) .* bo_u(op, 1, 0, 1)
        return combo * C
    elseif offset == 1
        denom = 2l + 3
        C = sqrt(max((l + 1)^2 - m^2, 0)) * l * (l + 2) / denom
        combo = -(l + 3) .* bo_u(op, 0, 0, 0)
        combo .+= -(l + 1) .* bo_u(op, 1, 1, 0)
        combo .+= -2 .* bo_u(op, 1, 0, 1)
        return combo * C
    elseif offset == 2
        denom = (3 + 2l) * (5 + 2l)
        sqrt1 = sqrt(max((2 + l - m) * (1 + l + m), 0))
        sqrt2 = sqrt(max((1 + l - m) * (2 + l + m), 0))
        C = 3 * l * (l + 3) * sqrt1 * sqrt2 / denom
        combo = -(l + 5) .* bo_u(op, 0, 0, 0)
        combo .+= -3 .* bo_u(op, 1, 0, 1)
        combo .+= -(l + 2) .* bo_u(op, 1, 1, 0)
        return combo * C
    else
        return zeros(ComplexF64, size(bo_u(op, 0, 0, 0)))
    end
end

function induction_f_utor_expected(op, l, m, offset)
    term = bo_u(op, 1, 0, 0)
    if offset == -1
        denom = 1 - 2l
        C = 18im * m * sqrt(max(l^2 - m^2, 0)) / denom
        return C .* term
    elseif offset == 0
        return -2im * m .* term
    elseif offset == 1
        denom = 3 + 2l
        C = -18im * m * sqrt(max((l + 1)^2 - m^2, 0)) / denom
        return C .* term
    else
        return zeros(ComplexF64, size(term))
    end
end

function induction_g_upol_expected(op, l, m, offset)
    if offset == -1
        denom = 2l - 1
        C = 3im * m * sqrt(max(l^2 - m^2, 0)) / denom
        combo = -2 * (-3 + l) .* bo_u(op, 0, 1, 0)
        combo .+= -2 * (-3 + l) .* bo_u(op, 0, 0, 1)
        combo .+= -2 * (3 + l^2) .* bo_u(op, -1, 0, 0)
        combo .+= 6 .* bo_u(op, 1, 0, 2)
        combo .+= -2 * (-3 + l) .* bo_u(op, 1, 1, 1)
        combo .+= ( -1 + l) * l .* bo_u(op, 1, 2, 0)
        return C .* combo
    elseif offset == 1
        denom = 2l + 3
        C = 3im * m * sqrt(max((l + 1)^2 - m^2, 0)) / denom
        combo = 2 * (4 + l) .* bo_u(op, 0, 1, 0)
        combo .+= 2 * (4 + l) .* bo_u(op, 0, 0, 1)
        combo .+= -2 * (4 + 2l + l^2) .* bo_u(op, -1, 0, 0)
        combo .+= 6 .* bo_u(op, 1, 0, 2)
        combo .+= (2 + 3l + l^2) .* bo_u(op, 1, 2, 0)
        combo .+= 2 * (4 + l) .* bo_u(op, 1, 1, 1)
        return C .* combo
    else
        return zeros(ComplexF64, size(bo_u(op, 0, 0, 0)))
    end
end

function induction_g_utor_expected(op, l, m, offset)
    if offset == -2
        denom = 3 - 8l + 4l^2
        C = 3 * (l - 2) * (l + 1) * sqrt(max((l - m) * (-1 + l + m) * (-1 + l - m) * (l + m), 0)) / denom
        combo = (-4 + l) .* bo_u(op, 0, 0, 0)
        combo .+= -3 .* bo_u(op, 1, 0, 1)
        combo .+= (-1 + l) .* bo_u(op, 1, 1, 0)
        return combo * C
    elseif offset == -1
        denom = 2l - 1
        C = sqrt(max((l - m) * (l + m), 0)) * (l^2 - 1) / denom
        combo = (l - 2) .* bo_u(op, 0, 0, 0)
        combo .+= l .* bo_u(op, 1, 1, 0)
        combo .+= -2 .* bo_u(op, 1, 0, 1)
        return combo * C
    elseif offset == 0
        C = 3 * (l + l^2 - 3m^2) / (-3 + 4l * (1 + l))
        combo = (6 - l - l^2) .* bo_u(op, 0, 0, 0)
        combo .+= (l + l^2) .* bo_u(op, 1, 1, 0)
        combo .+= -2 * (-3 + l + l^2) .* bo_u(op, 1, 0, 1)
        return combo * C
    elseif offset == 1
        denom = 2l + 3
        C = -sqrt(max((l + m + 1) * (l + 1 - m), 0)) * l * (l + 2) / denom
        combo = (l + 3) .* bo_u(op, 0, 0, 0)
        combo .+= (l + 1) .* bo_u(op, 1, 1, 0)
        combo .+= 2 .* bo_u(op, 1, 0, 1)
        return combo * C
    elseif offset == 2
        denom = (3 + 2l) * (5 + 2l)
        C = 3 * l * (l + 3) * sqrt(max((2 + l - m) * (1 + l + m) * (1 + l - m) * (2 + l + m), 0)) / denom
        combo = -(5 + l) .* bo_u(op, 0, 0, 0)
        combo .+= -3 .* bo_u(op, 1, 0, 1)
        combo .+= -(2 + l) .* bo_u(op, 1, 1, 0)
        return combo * C
    else
        return zeros(ComplexF64, size(bo_u(op, 0, 0, 0)))
    end
end

function magnetic_diffusion_f_expected(op, l)
    Em = op.params.E / op.params.Pm
    L = l * (l + 1)
    combo = -L .* dense(op.r0_D0_f) .+ 2 .* dense(op.r1_D1_f) .+ dense(op.r2_D2_f)
    return Em .* combo
end

function magnetic_diffusion_g_expected(op, l)
    Em = op.params.E / op.params.Pm
    L = l * (l + 1)
    combo = -L .* dense(op.r0_D0_g) .+ 2 .* dense(op.r1_D1_g) .+ dense(op.r2_D2_g)
    return Em * L .* combo
end

function b_poloidal_expected(op, l)
    L = l * (l + 1)
    return L .* dense(op.r2_D0_f)
end

function b_toroidal_expected(op, l)
    L = l * (l + 1)
    return L .* dense(op.r2_D0_g)
end

op = setup_operator()
m = op.params.m

@testset "Lorentz poloidal axial" begin
    for l in op.ll_u
        diag_actual = dense(operator_lorentz_poloidal_diagonal(op, l, op.params.Le))
        diag_expect = lorentz_upol_expected(op, l, m, 0)
        @test normdiff(diag_actual, diag_expect) ≤ TOL
        for offset in (-2:-1) ∪ (1:2)
            if offset < 0 && l + offset < minimum(op.ll_u)
                continue
            end
            actual = dense(operator_lorentz_poloidal_offdiag(op, l, m, offset, op.params.Le)[1])
            expect = lorentz_upol_expected(op, l, m, offset)
            @test normdiff(actual, expect) ≤ TOL
        end
    end
end

@testset "Lorentz toroidal axial" begin
    for l in op.ll_v
        actual_diag = dense(operator_lorentz_toroidal(op, l, op.params.Le))
        expect_diag = lorentz_utor_expected(op, l, m)
        @test normdiff(actual_diag, expect_diag) ≤ TOL
        for offset in -2:2
            actual, offd = operator_lorentz_poloidal_offdiag(op, l, m, offset, op.params.Le)
        end
    end
end

