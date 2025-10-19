using Cross

println("Quick test of corrected implementation...")

# Simple case
E = 1e-4
Pr = 1.0
χ = 0.35
m = 10
lmax = 15
Nr = 16
Ra = 1e7

println("Creating operator...")
params = OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=m, lmax=lmax, Nr=Nr,
                     mechanical_bc=:no_slip, thermal_bc=:fixed_temperature,
                     use_kore_weighting=true)

op = LinearStabilityOperator(params)
println("Operator created successfully!")
println("Total DOF: ", op.total_dof)

println("\nTrying eigenvalue solve...")
try
    σ, ω, vec = find_growth_rate(op)
    println("SUCCESS!")
    println("Growth rate σ = ", σ)
    println("Drift frequency ω = ", ω)
catch e
    println("ERROR: ", e)
    showerror(stdout, e, catch_backtrace())
end
