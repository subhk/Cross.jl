#!/usr/bin/env julia
#
# Demonstration of the linear stability solver derived from Equations (10)–(19)
# in docs/Onset_convection.pdf.  The script fixes the Rayleigh number and
# computes the leading complex growth rate (σ + iω) for a rotating spherical
# shell using the parameters from Figure 2 of the reference.

repo_root = normpath(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, repo_root)

# Allow overriding the SHTnsKit location through an environment variable
sht_local = get(ENV, "SHTNSKIT_PATH", joinpath(repo_root, "..", "SHTnsKit.jl"))
if isdir(sht_local)
    push!(LOAD_PATH, joinpath(sht_local, "src"))
else
    @warn "SHTnsKit path not found" sht_local
end

# Prefer a locally checked-out FeastKit before falling back to the registered version.
feast_candidates = String[]
if haskey(ENV, "FEASTKIT_PATH")
    push!(feast_candidates, ENV["FEASTKIT_PATH"])
end
push!(feast_candidates, joinpath(repo_root, "..", "FeastKit.jl"))
push!(feast_candidates, joinpath(repo_root, "..", "Feast.jl"))

feast_path = nothing
for candidate in feast_candidates
    if isdir(candidate)
        feast_path = candidate
        push!(LOAD_PATH, joinpath(candidate, "src"))
        break
    end
end

if feast_path === nothing
    @warn "FeastKit path not found; falling back to the registered package" feast_candidates
end

using Cross
using Printf

E = 1e-5
Pr = 1.0
Ra = 2.1e7
ri = 0.35
ro = 1.0
Nr = 64

meridional_points = 96  # choose independent meridional resolution (θ collocation)

println("m    Re(λ₁)          Im(λ₁)          iterations")
println("------------------------------------------------")

for m in 1:20
    lmax = max(48, m + 6)
    params = ShellParams(m=m, E=E, Pr=Pr, Ra=Ra, ri=ri, ro=ro, lmax=lmax, Nr=Nr)
    try
        vals, _, _, info = leading_modes(params; nθ=meridional_points, nev=2, which=:LR, tol=1e-6, maxiter=120)
        λ1 = vals[1]
        @printf("%2d  %12.5e  %12.5e  %5d\n", m, real(λ1), imag(λ1), info.iterations)
    catch err
        @printf("%2d  %12s  %12s      --\n", m, "ERROR", "ERROR")
        @warn "Failed to converge" m err
    end
end
