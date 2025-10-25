# =============================================================================
#  Eigenvalue Solver for Onset of Convection
#
#  Solves the generalized eigenvalue problem:
#      A·x = σ·B·x
#
#  where σ = σ_r + iω with:
#  - σ_r: growth rate (σ_r = 0 at onset)
#  - ω: drift frequency
#
#  References:
#  - Barik et al. (2023), Earth and Space Science
#  - Dormy et al. (2004), Journal of Fluid Mechanics
# =============================================================================

module OnsetEigenvalueSolver

using LinearAlgebra
using SparseArrays
using Arpack
using Printf

export solve_eigenvalue_problem,
       find_critical_rayleigh,
       find_onset_parameters

"""
    solve_eigenvalue_problem(A, B; nev=20, sigma=nothing, which=:LR,
                             selection=:maxreal, tol=1e-10)

Solve the generalized eigenvalue problem A·x = σ·B·x for sparse matrices.

# Arguments
- `A::SparseMatrixCSC`: Operator matrix (physics terms)
- `B::SparseMatrixCSC`: Mass matrix (time derivative weights)
- `nev::Int=20`: Number of eigenvalues to compute
- `sigma::Union{Nothing,Number}=nothing`: Shift for shift-invert mode (set to a
  numeric value to target eigenvalues near that shift)
- `which::Symbol=:LR`: Which eigenvalues to compute (see `Arpack.eigs`)
- `selection::Symbol=:maxreal`: How to order the returned eigenvalues:
  - `:maxreal`: sort by descending real part
  - `:minabs`: sort by ascending magnitude
  - `:closest_real`: sort by ascending |Re(σ)|
- `tol::Float64=1e-10`: Convergence tolerance

# Returns
- `eigenvalues::Vector{ComplexF64}`: Computed eigenvalues σ = σ_r + iω
- `eigenvectors::Matrix{ComplexF64}`: Corresponding eigenvectors
- `info::Dict`: Information about the solve

# Notes
For onset of convection problems it is often helpful to use shift-invert with
`sigma=0.0`, `which=:LM`, and `selection=:closest_real` to isolate the mode
closest to neutral stability.
"""
function solve_eigenvalue_problem(A::SparseMatrixCSC, B::SparseMatrixCSC;
                                 nev::Int=20,
                                 sigma::Union{Nothing,Number}=nothing,
                                 which::Symbol=:LR,
                                 selection::Symbol=:maxreal,
                                 tol::Float64=1e-10,
                                 maxiter::Int=1000)

    n = size(A, 1)
    @assert size(A) == size(B) "A and B must have same dimensions"
    @assert size(A, 1) == size(A, 2) "Matrices must be square"

    println("Solving eigenvalue problem...")
    println("  Matrix size: $n × $n")
    println("  A nnz: $(nnz(A)), B nnz: $(nnz(B))")
    println("  Computing $nev eigenvalues with which=$which")

    try
        # Use Arpack's eigs for sparse generalized eigenvalue problem
        # We solve (A - σB)^{-1}·B·x = θ·x where σ = eigenvalue

        n_eigs = clamp(nev, 1, max(n - 2, 1))
        if sigma === nothing
            eigenvalues, eigenvectors, nconv, niter, nmult, resid = eigs(
                A, B;
                nev = n_eigs,
                which = which,
                tol = tol,
                maxiter = maxiter
            )
        else
            eigenvalues, eigenvectors, nconv, niter, nmult, resid = eigs(
                A, B;
                nev = n_eigs,
                which = which,
                sigma = sigma,
                tol = tol,
                maxiter = maxiter
            )
        end

        select_values = real.(eigenvalues)
        perm = nothing

        if selection == :maxreal
            perm = sortperm(select_values, rev=true)
        elseif selection == :minabs
            perm = sortperm(abs.(eigenvalues))
        elseif selection == :closest_real
            perm = sortperm(abs.(select_values))
        else
            error("Unknown selection strategy $(selection)")
        end

        eigenvalues = eigenvalues[perm]
        eigenvectors = eigenvectors[:, perm]

        println("  ✓ Converged: $nconv eigenvalues in $niter iterations")
        println("  Selected mode: σ = $(eigenvalues[1]) using $selection selection")
        println("    Growth rate: σ_r = $(real(eigenvalues[1]))")
        println("    Drift frequency: ω = $(imag(eigenvalues[1]))")

        info = Dict(
            "nconv" => nconv,
            "niter" => niter,
            "nmult" => nmult,
            "residual" => resid,
            "selected" => eigenvalues[1],
            "selection" => selection
        )

        return eigenvalues, eigenvectors, info

    catch err
        @error "Eigenvalue solver failed" exception=(err, catch_backtrace())
        rethrow()
    end
end

"""
    find_critical_rayleigh(operator_builder, E, χ, m;
                          Ra_min=1e4, Ra_max=1e10,
                          tol=1e-6, growth_tol=1e-6, max_iter=50)

Find the critical Rayleigh number Ra_c where the growth rate σ_r = 0.

Uses a safeguarded Brent root finder (inverse quadratic interpolation with
bisection fallback) to mirror the strategy used in Kore.

# Arguments
- `operator_builder::Function`: Function that takes Ra and returns (A, B)
- `E::Float64`: Ekman number
- `χ::Float64`: Radius ratio
- `m::Int`: Azimuthal wavenumber
- `Ra_min::Float64`: Lower bracket for Ra
- `Ra_max::Float64`: Upper bracket for Ra
- `tol::Float64`: Relative tolerance on Ra (controls absolute tolerance internally)
- `growth_tol::Float64`: Absolute tolerance on the residual growth rate
- `max_iter::Int`: Maximum number of iterations

# Returns
- `Ra_c::Float64`: Critical Rayleigh number
- `ω_c::Float64`: Drift frequency at onset
- `σ_c::ComplexF64`: Full eigenvalue at onset
- `iterations::Int`: Number of iterations required
"""
function find_critical_rayleigh(operator_builder::Function, E::Float64, χ::Float64, m::Int;
                               Ra_min::Float64=1e4, Ra_max::Float64=1e10,
                               tol::Float64=1e-6, growth_tol::Float64=1e-6,
                               max_iter::Int=50, nev::Int=10)

    println("\n" * "="^80)
    println("Finding Critical Rayleigh Number")
    println("="^80)
    println("Parameters: E = $E, χ = $χ, m = $m")
    println("Bracket: [$Ra_min, $Ra_max]")
    println("Tolerance: $tol (relative), growth_tol: $growth_tol (|σ_r|)")
    println()

    """Get growth rate for a given Ra"""
    function growth_rate(Ra)
        println("  Testing Ra = $(Ra)...")
        A, B = operator_builder(Ra)

        # Find mode closest to neutral stability using shift-invert targeting σ≈0
        solver_nev = max(nev, 6)
        eigenvalues, _, info = solve_eigenvalue_problem(
            A, B;
            nev = solver_nev,
            sigma = 0.0,
            which = :LM,
            selection = :closest_real
        )

        σ = eigenvalues[1]
        σ_r = real(σ)

        println("    → Growth rate: σ_r = $(σ_r)")
        return σ_r, σ
    end

    # Cache evaluations to avoid redundant solves when scanning
    known_values = Dict{Float64, Tuple{Float64, ComplexF64}}()

    function growth_rate_cached(Ra)
        Ra_key = Float64(Ra)
        if haskey(known_values, Ra_key)
            return known_values[Ra_key]
        end
        σ_r, σ = growth_rate(Ra)
        known_values[Ra_key] = (σ_r, σ)
        return σ_r, σ
    end

    # Initial bracket check
    println("Checking initial bracket...")
    σ_r_min, σ_min = growth_rate_cached(Ra_min)
    if abs(σ_r_min) < growth_tol
        println("Lower bracket already satisfies growth tolerance.")
        return Ra_min, imag(σ_min), σ_min, 0
    end

    σ_r_max, σ_max = growth_rate_cached(Ra_max)
    if abs(σ_r_max) < growth_tol
        println("Upper bracket already satisfies growth tolerance.")
        return Ra_max, imag(σ_max), σ_max, 0
    end

    if σ_r_min * σ_r_max > 0
        @warn "Initial bracket may not contain critical Ra" σ_r_min σ_r_max
        max_bracket_expansions = 12
        expansion_iter = 0
        while σ_r_min * σ_r_max > 0 && expansion_iter < max_bracket_expansions
            expansion_iter += 1
            if σ_r_min > 0 && σ_r_max > 0
                Ra_min /= 2.0
                println("  Expansion $expansion_iter: lowering Ra_min → $Ra_min")
                if Ra_min <= 0
                    error("Lower Rayleigh bound reached non-positive value while trying to bracket root.")
                end
                σ_r_min, σ_min = growth_rate_cached(Ra_min)
            elseif σ_r_min < 0 && σ_r_max < 0
                Ra_max *= 2.0
                println("  Expansion $expansion_iter: raising Ra_max → $Ra_max")
                σ_r_max, σ_max = growth_rate_cached(Ra_max)
            else
                break
            end
        end
        if σ_r_min * σ_r_max > 0
            println("  Expansion attempts exhausted. Performing logarithmic scan for sign change...")
            min_scan = max(Ra_min, 10.0) / 10.0
            max_scan = Ra_max * 10.0
            if min_scan <= 0
                min_scan = tol
            end
            scan_points = 30
            scan_values = exp10.(LinRange(log10(min_scan), log10(max_scan), scan_points))
            scan_values = sort(unique(vcat(Ra_min, Ra_max, scan_values)))

            bracket_found = false
            last_ra = nothing
            lastσ_r = 0.0
            lastσ = 0.0 + 0.0im

            for Ra in scan_values
                σ_r, σ = growth_rate_cached(Ra)
                if last_ra !== nothing && σ_r * lastσ_r <= 0
                    println("  Found sign change between $(last_ra) and $(Ra) during scan.")
                    Ra_min, Ra_max = last_ra, Ra
                    σ_r_min, σ_min = lastσ_r, lastσ
                    σ_r_max, σ_max = σ_r, σ
                    bracket_found = true
                    break
                end
                last_ra = Ra
                lastσ_r = σ_r
                lastσ = σ
            end

            if !bracket_found
                error("Unable to bracket the critical Rayleigh number: growth rate has same sign across scanned range.")
            end
        end
    end

    # Safeguarded Brent search (closely follows implementations in literature)
    println("\nStarting Brent search...")
    Ra_a, Ra_b, Ra_c = Ra_min, Ra_max, Ra_min
    σ_r_a, σ_r_b, σ_r_c = σ_r_min, σ_r_max, σ_r_min
    σ_a, σ_b, σ_c = σ_min, σ_max, σ_min

    if σ_r_a * σ_r_b >= 0
        error("Brent search requires opposite signs at the bracket endpoints.")
    end

    abs_tol = tol * max(abs(Ra_a), abs(Ra_b), 1.0)
    d = Ra_b - Ra_a
    e = d

    for iter in 1:max_iter
        if (σ_r_b > 0 && σ_r_c > 0) || (σ_r_b < 0 && σ_r_c < 0)
            Ra_c = Ra_a
            σ_r_c = σ_r_a
            σ_c = σ_a
            d = Ra_b - Ra_a
            e = d
        end

        if abs(σ_r_c) < abs(σ_r_b)
            Ra_a, Ra_b, Ra_c = Ra_b, Ra_c, Ra_b
            σ_r_a, σ_r_b, σ_r_c = σ_r_b, σ_r_c, σ_r_b
            σ_a, σ_b, σ_c = σ_b, σ_c, σ_b
        end

        tol_act = 2 * eps(abs(Ra_b)) + abs_tol
        m = 0.5 * (Ra_c - Ra_b)

        println("\nIteration $iter:")
        println("  Bracket: [$(Ra_a), $(Ra_c)] with current Ra = $(Ra_b)")
        println("  Growth rates: f(a)=$(σ_r_a), f(b)=$(σ_r_b), f(c)=$(σ_r_c)")

        if abs(m) <= tol_act || σ_r_b == 0
            println("\n" * "="^80)
            println("✓ CONVERGED")
            println("="^80)
            Ra_c_final = Ra_b
            ω_c = imag(σ_b)
            println("Critical Rayleigh number: Ra_c = $(Ra_c_final)")
            println("Drift frequency: ω_c = $(ω_c)")
            println("Residual growth rate: σ_r = $(σ_r_b)")
            println("Iterations: $iter")
            println("="^80)
            return Ra_c_final, ω_c, σ_b, iter
        end

        if abs(e) < tol_act || abs(σ_r_a) <= abs(σ_r_b)
            d = m
            e = m
            println("  Using bisection step.")
        else
            s = σ_r_b / σ_r_a
            if Ra_a == Ra_c
                # Secant method
                p = 2 * m * s
                q = 1 - s
            else
                q = σ_r_a / σ_r_c
                r = σ_r_b / σ_r_c
                p = s * (2 * m * q * (q - r) - (Ra_b - Ra_a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            end

            if p > 0
                q = -q
            else
                p = -p
            end

            if (2p < 3m * q - abs(tol_act * q)) && (p < abs(0.5 * e * q))
                e = d
                d = p / q
                println("  Using inverse interpolation step.")
            else
                d = m
                e = m
                println("  Interpolation rejected, falling back to bisection.")
            end
        end

        Ra_a = Ra_b
        σ_r_a = σ_r_b
        σ_a = σ_b

        if abs(d) > tol_act
            Ra_b += d
        else
            Ra_b += m >= 0 ? tol_act : -tol_act
        end

        σ_r_b, σ_b = growth_rate_cached(Ra_b)

        if abs(σ_r_b) < growth_tol
            println("\n" * "="^80)
            println("✓ CONVERGED")
            println("="^80)
            ω_c = imag(σ_b)
            println("Critical Rayleigh number: Ra_c = $(Ra_b)")
            println("Drift frequency: ω_c = $(ω_c)")
            println("Residual growth rate: σ_r = $(σ_r_b)")
            println("Iterations: $iter")
            println("="^80)
            return Ra_b, ω_c, σ_b, iter
        end
    end

    @warn "Maximum iterations reached without convergence (Brent)"
    println("Returning best current estimate.")
    ω_c = imag(σ_b)
    return Ra_b, ω_c, σ_b, max_iter
end

"""
    find_onset_parameters(params_template, m_range; kwargs...)

Find onset parameters (Ra_c, m_c, ω_c) by scanning over azimuthal wavenumbers.

# Arguments
- `params_template::SparseOnsetParams`: Template parameters (E, χ, Pr, etc.)
- `m_range::AbstractVector{Int}`: Range of m values to test
- `kwargs...`: Passed to find_critical_rayleigh

# Returns
- `Ra_c::Float64`: Critical Rayleigh number
- `m_c::Int`: Critical azimuthal wavenumber
- `ω_c::Float64`: Critical drift frequency
- `results::Dict`: Full results for all m values tested
"""
function find_onset_parameters(operator_builder_factory::Function,
                               E::Float64, χ::Float64, Pr::Float64,
                               m_range::AbstractVector{Int};
                               kwargs...)

    println("\n" * "="^80)
    println("SCANNING FOR ONSET PARAMETERS")
    println("="^80)
    println("E = $E, χ = $χ, Pr = $Pr")
    println("Testing m ∈ $m_range")
    println("="^80)

    results = Dict()
    Ra_c_min = Inf
    m_c = 0
    ω_c_best = 0.0

    for m in m_range
        println("\n" * "~"^80)
        println("Testing m = $m")
        println("~"^80)

        try
            # Create operator builder for this m
            operator_builder = operator_builder_factory(E, χ, Pr, m)

            # Find critical Ra for this m
            Ra_c, ω_c, σ_c, iters = find_critical_rayleigh(
                operator_builder, E, χ, m; kwargs...
            )

            results[m] = (Ra_c=Ra_c, ω_c=ω_c, σ_c=σ_c, iters=iters)

            # Track minimum
            if Ra_c < Ra_c_min
                Ra_c_min = Ra_c
                m_c = m
                ω_c_best = ω_c
            end

            println("Result for m=$m: Ra_c = $(Ra_c), ω_c = $(ω_c)")

        catch err
            @warn "Failed for m=$m" exception=err
            results[m] = (error=err,)
        end
    end

    println("\n" * "="^80)
    println("ONSET PARAMETERS FOUND")
    println("="^80)
    println("Critical Rayleigh number: Ra_c = $(Ra_c_min)")
    println("Critical azimuthal wavenumber: m_c = $(m_c)")
    println("Critical drift frequency: ω_c = $(ω_c_best)")
    println("="^80)

    return Ra_c_min, m_c, ω_c_best, results
end

end  # module OnsetEigenvalueSolver
