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
                  --solver=arpack|krylov         Choose eigen solver (default arpack)
                  --arpack-shift=<real+imag>     Optional shift for Arpack (e.g. 0.1+0.1im)
                  --theta-points=<int>           Number of meridional grid points (default 96)
                  --help                         Show this message
                """)
            exit(0)
        elseif startswith(arg, "--solver=")
            opts[:solver] = Symbol(lowercase(split(arg, '=' )[2]))
        elseif startswith(arg, "--arpack-shift=")
            raw = split(arg, '=' )[2]
            shift_val = try
                parse(ComplexF64, raw)
            catch
                parse(Float64, raw) + 0im
            end
            opts[:arpack_shift] = shift_val
        elseif startswith(arg, "--theta-points=")
            opts[:theta_points] = parse(Int, split(arg, '=' )[2])
        else
            @warn "Ignoring unrecognised argument" arg
        end
    end
    return opts
end

cli_opts = parse_cli_args(ARGS)

parse_shift_string(str) = try
    parse(ComplexF64, str)
catch
    parse(Float64, str) + 0im
end

function solver_iterations(info)
    info === nothing && return missing
    if hasproperty(info, :niter)
        return getproperty(info, :niter)
    elseif hasproperty(info, :numiter)
        return getproperty(info, :numiter)
    elseif hasproperty(info, :iterations)
        return getproperty(info, :iterations)
    else
        return missing
    end
end

solver = get(cli_opts, :solver, Symbol(lowercase(get(ENV, "CROSS_SOLVER", "arpack"))))
if !(solver in (:arpack, :krylov))
    @warn "Unknown solver requested; defaulting to :arpack" solver
    solver = :arpack
end

arpack_shift = if haskey(cli_opts, :arpack_shift)
    cli_opts[:arpack_shift]
elseif haskey(ENV, "CROSS_ARPACK_SHIFT")
    parse_shift_string(ENV["CROSS_ARPACK_SHIFT"])
else
    nothing
end

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
                                         arpack_shift=arpack_shift)
        λ1 = vals[1]
        iter_val = solver_iterations(info)
        iter_str = iter_val isa Integer ? @sprintf("%5d", iter_val) : "    --"
        @printf("%2d  %12.5e  %12.5e  %s\n", m, real(λ1), imag(λ1), iter_str)
    catch err
        @printf("%2d  %12s  %12s      --\n", m, "ERROR", "ERROR")
        @warn "Failed to converge" m err exception=(err, catch_backtrace())
    end
end
