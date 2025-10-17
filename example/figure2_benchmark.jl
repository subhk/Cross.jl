#!/usr/bin/env julia
#
# Benchmark script reproducing the neutral-stability curve shown in
# Figure 2 of docs/Onset_convection.pdf.  For each azimuthal order `m` it
# performs a simple bisection search on the Rayleigh number to find the
# value at which the leading growth rate crosses zero (σ = 0).  The script
# assumes no-slip, fixed-temperature boundaries exactly as in the paper.

repo_root = normpath(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, repo_root)

# Allow overriding the SHTnsKit location through an environment variable.
sht_local = get(ENV, "SHTNSKIT_PATH", joinpath(repo_root, "..", "SHTnsKit.jl"))
if isdir(sht_local)
    push!(LOAD_PATH, joinpath(sht_local, "src"))
else
    @warn "SHTnsKit path not found" sht_local
end

using Cross
using Printf

const E = 1e-5
const Pr = 1.0
const ri = 0.35
const ro = 1.0
const Nr = 64
const nθ = 96

# Rayleigh number brackets used for the bisection searches.  The values
# were chosen to cover the neutral curve reported in the paper.
const RA_LOWER = 1.5e7
const RA_UPPER = 3.0e7
const BIS_TOL  = 1e-3
const MAX_ITERS = 40

"""
    leading_growth_rate(m, Ra; lmax, nθ)

Return the largest real part σ of the eigenvalues for the specified `m`
and Rayleigh number `Ra`.
"""
function leading_growth_rate(m::Int, Ra::Float64; lmax::Int, nθ::Int)
    params = ShellParams(m=m, E=E, Pr=Pr, Ra=Ra,
                         ri=ri, ro=ro, lmax=lmax, Nr=Nr)
    vals, _, _, _ = leading_modes(params; nθ=nθ, nev=4,
                                  which=:LR, tol=1e-6, maxiter=120)
    return maximum(real, vals)
end

"""
    critical_Ra(m; bracket=(RA_LOWER, RA_UPPER))

Compute the neutral Rayleigh number for azimuthal order `m` by bisection.
"""
function critical_Ra(m::Int; bracket=(RA_LOWER, RA_UPPER))
    lmax = max(48, m + 6)
    Ra_low, Ra_high = float.(bracket)
    σ_low = leading_growth_rate(m, Ra_low; lmax=lmax, nθ=nθ)
    σ_high = leading_growth_rate(m, Ra_high; lmax=lmax, nθ=nθ)

    if σ_low > 0
        error("Lower bracket Ra=$(Ra_low) already unstable (σ=$(σ_low)) for m=$m.")
    elseif σ_high < 0
        error("Upper bracket Ra=$(Ra_high) still stable (σ=$(σ_high)) for m=$m.")
    end

    for iter in 1:MAX_ITERS
        Ra_mid = 0.5 * (Ra_low + Ra_high)
        σ_mid = leading_growth_rate(m, Ra_mid; lmax=lmax, nθ=nθ)

        if abs(σ_mid) < 1e-6 || abs(Ra_high - Ra_low)/Ra_mid < BIS_TOL
            return Ra_mid, σ_mid, iter
        end

        if σ_mid > 0
            Ra_high = Ra_mid
            σ_high = σ_mid
        else
            Ra_low = Ra_mid
            σ_low = σ_mid
        end
    end

    return 0.5 * (Ra_low + Ra_high), 0.5 * (σ_low + σ_high), MAX_ITERS
end

println("Benchmarking neutral curve at E=$(E), Pr=$(Pr)")
println("m    Rac             sigma        iterations")
println("------------------------------------------------")

results = Dict{Int, Tuple{Float64,Float64,Int}}()

for m in 1:20
    try
        Rac, σ, iters = critical_Ra(m)
        results[m] = (Rac, σ, iters)
        @printf("%2d  %12.5e  %12.5e  %5d\n", m, Rac, σ, iters)
    catch err
        @printf("%2d  %12s  %12s      --\n", m, "ERROR", "ERROR")
        @warn "Failed to determine Rac" m err
    end
end

println("\nTip: compare the Rac values above with Figure 2 of the paper.")
println("     Adjust `RA_LOWER`/`RA_UPPER` if any mode fails to bracket the root.")
