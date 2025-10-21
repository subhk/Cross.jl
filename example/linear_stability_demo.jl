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

# ------------------------------------------------------------------------------
# Solver configuration: environment variables or CLI flags
# ------------------------------------------------------------------------------

function parse_cli_args(args)
    opts = Dict{Symbol,Any}()
    for arg in args
        if arg in ("-h", "--help")
            println("""
                Usage: julia example/linear_stability_demo.jl [options]

                Options:
                  --solver=feast|krylov          Choose eigen solver (default feast)
                  --feast-center=<real>+<imag>im Complex center for FEAST contour (default 0.0+0.0im)
                  --feast-radius=<float>         FEAST contour radius (default 1.0)
                  --feast-M0=<int>               FEAST subspace size (default 48)
                  --theta-points=<int>           Number of meridional grid points (default 96)
                  --help                         Show this message
                """)
            exit(0)
        elseif startswith(arg, "--solver=")
            opts[:solver] = Symbol(lowercase(split(arg, '=' )[2]))
        elseif startswith(arg, "--feast-center=")
            val = split(arg, '=' )[2]
            center_val = try
                parse(ComplexF64, val)
            catch
                parse(Float64, val) + 0im
            end
            opts[:feast_center] = center_val
        elseif startswith(arg, "--feast-radius=")
            opts[:feast_radius] = parse(Float64, split(arg, '=' )[2])
        elseif startswith(arg, "--feast-M0=")
            opts[:feast_M0] = parse(Int, split(arg, '=' )[2])
        elseif startswith(arg, "--theta-points=")
            opts[:theta_points] = parse(Int, split(arg, '=' )[2])
        else
            @warn "Ignoring unrecognised argument" arg
        end
    end
    return opts
end

cli_opts = parse_cli_args(ARGS)

solver = get(cli_opts, :solver, Symbol(lowercase(get(ENV, "CROSS_SOLVER", "feast"))))
if !(solver in (:feast, :krylov))
    @warn "Unknown solver requested; defaulting to :feast" solver
    solver = :feast
end
feast_center = get(cli_opts, :feast_center,
                   parse(Float64, get(ENV, "CROSS_FEAST_CENTER_REAL", "0.0")) +
                   parse(Float64, get(ENV, "CROSS_FEAST_CENTER_IMAG", "0.0")) * im)
feast_radius = get(cli_opts, :feast_radius, parse(Float64, get(ENV, "CROSS_FEAST_RADIUS", "1.0")))
feast_M0 = get(cli_opts, :feast_M0, parse(Int, get(ENV, "CROSS_FEAST_M0", "48")))
meridional_points = get(cli_opts, :theta_points, parse(Int, get(ENV, "CROSS_THETA_POINTS", "96")))

println("m    Re(λ₁)          Im(λ₁)          iterations")
println("------------------------------------------------")

for m in 1:20
    lmax = max(48, m + 6)
    params = ShellParams(m=m, E=E, Pr=Pr, Ra=Ra, ri=ri, ro=ro, lmax=lmax, Nr=Nr)
    try
        vals, _, _, info = leading_modes(params;
                                         nθ=meridional_points,
                                         nev=2,
                                         which=:LR,
                                         tol=1e-6,
                                         maxiter=120,
                                         solver=solver,
                                         feast_center=feast_center,
                                         feast_radius=feast_radius,
                                         feast_M0=feast_M0)
        λ1 = vals[1]
        @printf("%2d  %12.5e  %12.5e  %5d\n", m, real(λ1), imag(λ1), info.iterations)
    catch err
        @printf("%2d  %12s  %12s      --\n", m, "ERROR", "ERROR")
        @warn "Failed to converge" m err
    end
end
