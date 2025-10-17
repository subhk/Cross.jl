#!/usr/bin/env julia
#
# Demonstration of the linear stability solver derived from Equations (10)–(19)
# in docs/Onset_convection.pdf.  The script computes the leading Rayleigh
# number eigenvalue for a rotating spherical shell with the same parameters used
# in Figure 2 of the reference.

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

E = 1e-5
Pr = 1.0
ri = 0.35
ro = 1.0
Nr = 64

meridional_points = 96  # choose independent meridional resolution (θ collocation)

println("m    Ra₁            Im(Ra₁)        iterations")
println("------------------------------------------------")

for m in 1:20
    lmax = max(48, m + 6)
    params = ShellParams(m=m, E=E, Pr=Pr, ri=ri, ro=ro, lmax=lmax, Nr=Nr)
    try
        onset = critical_rayleigh(params; nθ=meridional_points, nev=6, tol=1e-6, maxiter=120)
        info = onset.info
        @printf("%2d  %12.5e  %12.5e  %5d\n", m, onset.Ra, imag(onset.λ), info.iterations)
    catch err
        @printf("%2d  %12s  %12s      --\n", m, "ERROR", "ERROR")
        @warn "Failed to converge" m err
    end
end
