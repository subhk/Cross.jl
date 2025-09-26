#!/usr/bin/env julia
#
# Demonstration of the linear stability solver derived from Equations (10)–(19)
# in docs/Onset_convection.pdf.  The script computes the dominant eigenvalue for
# a rotating spherical shell with the same parameters used in Figure 2 of the
# reference.

repo_root = normpath(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, repo_root)

# Allow overriding the SHTnsKit location through an environment variable
sht_local = get(ENV, "SHTNSKIT_PATH", joinpath(repo_root, "..", "SHTnsKit.jl"))
if isdir(sht_local)
    push!(LOAD_PATH, joinpath(sht_local, "src"))
else
    @warn "SHTnsKit path not found" sht_local
end

using Cross
using Printf

params = ShellParams(
    m = 13,
    E = 1e-5,
    Pr = 1.0,
    Ra = 2.1e7,
    ri = 0.35,
    ro = 1.0,
    lmax = 64,
    Nr = 80,
)

vals, vecs, op, info = leading_modes(params; nθ = params.lmax + 1, nev = 4, tol = 1e-6)

println("Leading eigenvalues for m=$(params.m):")
for (k, val) in enumerate(vals)
    @printf("  λ₍%d₎ = %12.5e + %12.5e im\n", k, real(val), imag(val))
end

σ = real(vals[1])
if σ > 0
    println("The chosen Rayleigh number is supercritical (σ > 0).")
elseif σ < 0
    println("The chosen Rayleigh number is subcritical (σ < 0).")
else
    println("Neutral stability detected (σ ≈ 0).")
end

println("Krylov iterations: ", info.iterations)
