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
    solve_eigenvalue_problem(A, B; nev=20, sigma=0.0, which=:LM, tol=1e-10)

Solve the generalized eigenvalue problem A·x = σ·B·x for sparse matrices.

# Arguments
- `A::SparseMatrixCSC`: Operator matrix (physics terms)
- `B::SparseMatrixCSC`: Mass matrix (time derivative weights)
- `nev::Int=20`: Number of eigenvalues to compute
- `sigma::Number=0.0`: Shift for shift-invert mode (targets eigenvalues near sigma)
- `which::Symbol=:LM`: Which eigenvalues to compute
  - `:LM`: Largest magnitude
  - `:SM`: Smallest magnitude
  - `:LR`: Largest real part
  - `:SR`: Smallest real part
  - `:LI`: Largest imaginary part
- `tol::Float64=1e-10`: Convergence tolerance

# Returns
- `eigenvalues::Vector{ComplexF64}`: Computed eigenvalues σ = σ_r + iω
- `eigenvectors::Matrix{ComplexF64}`: Corresponding eigenvectors
- `info::Dict`: Information about the solve

# Notes
For onset of convection problems, we typically want:
- `which=:LR` to find the most unstable mode (largest real part)
- `sigma=0.0` to target eigenvalues near the neutral stability curve
"""
function solve_eigenvalue_problem(A::SparseMatrixCSC, B::SparseMatrixCSC;
                                 nev::Int=20, sigma::Number=0.0,
                                 which::Symbol=:LR, tol::Float64=1e-10,
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

        if sigma == 0.0
            # Standard mode: find eigenvalues directly
            eigenvalues, eigenvectors, nconv, niter, nmult, resid = eigs(
                A, B;
                nev = min(nev, n-2),  # Need n-2 for Arpack
                which = which,
                tol = tol,
                maxiter = maxiter
            )
        else
            # Shift-invert mode: find eigenvalues near sigma
            eigenvalues, eigenvectors, nconv, niter, nmult, resid = eigs(
                A, B;
                nev = min(nev, n-2),
                which = which,
                sigma = sigma,
                tol = tol,
                maxiter = maxiter
            )
        end

        # Sort by real part (most unstable first)
        perm = sortperm(real.(eigenvalues), rev=true)
        eigenvalues = eigenvalues[perm]
        eigenvectors = eigenvectors[:, perm]

        println("  ✓ Converged: $nconv eigenvalues in $niter iterations")
        println("  Most unstable mode: σ = $(eigenvalues[1])")
        println("    Growth rate: σ_r = $(real(eigenvalues[1]))")
        println("    Drift frequency: ω = $(imag(eigenvalues[1]))")

        info = Dict(
            "nconv" => nconv,
            "niter" => niter,
            "nmult" => nmult,
            "residual" => resid,
            "most_unstable" => eigenvalues[1]
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
                          tol=1e-6, max_iter=50)

Find the critical Rayleigh number Ra_c where the growth rate σ_r = 0.

Uses bracketing and bisection to find the onset of convection.

# Arguments
- `operator_builder::Function`: Function that takes Ra and returns (A, B)
- `E::Float64`: Ekman number
- `χ::Float64`: Radius ratio
- `m::Int`: Azimuthal wavenumber
- `Ra_min::Float64`: Lower bracket for Ra
- `Ra_max::Float64`: Upper bracket for Ra
- `tol::Float64`: Tolerance for convergence (relative to Ra)
- `max_iter::Int`: Maximum number of iterations

# Returns
- `Ra_c::Float64`: Critical Rayleigh number
- `ω_c::Float64`: Drift frequency at onset
- `σ_c::ComplexF64`: Full eigenvalue at onset
- `iterations::Int`: Number of iterations required
"""
function find_critical_rayleigh(operator_builder::Function, E::Float64, χ::Float64, m::Int;
                               Ra_min::Float64=1e4, Ra_max::Float64=1e10,
                               tol::Float64=1e-6, max_iter::Int=50,
                               nev::Int=10)

    println("\n" * "="^80)
    println("Finding Critical Rayleigh Number")
    println("="^80)
    println("Parameters: E = $E, χ = $χ, m = $m")
    println("Bracket: [$Ra_min, $Ra_max]")
    println("Tolerance: $tol")
    println()

    """Get growth rate for a given Ra"""
    function growth_rate(Ra)
        println("  Testing Ra = $(Ra)...")
        A, B = operator_builder(Ra)

        # Find most unstable mode
        eigenvalues, _, info = solve_eigenvalue_problem(A, B; nev=nev, which=:LR)

        σ = eigenvalues[1]
        σ_r = real(σ)

        println("    → Growth rate: σ_r = $(σ_r)")
        return σ_r, σ
    end

    # Initial bracket check
    println("Checking initial bracket...")
    σ_r_min, σ_min = growth_rate(Ra_min)
    σ_r_max, σ_max = growth_rate(Ra_max)

    if σ_r_min * σ_r_max > 0
        @warn "Initial bracket may not contain critical Ra" σ_r_min σ_r_max
        if σ_r_min > 0
            # Both positive - need lower Ra
            Ra_min = Ra_min / 10.0
            println("Adjusting lower bound to $Ra_min...")
            σ_r_min, σ_min = growth_rate(Ra_min)
        else
            # Both negative - need higher Ra
            Ra_max = Ra_max * 10.0
            println("Adjusting upper bound to $Ra_max...")
            σ_r_max, σ_max = growth_rate(Ra_max)
        end
    end

    # Bisection search
    println("\nStarting bisection search...")
    Ra_a, Ra_b = Ra_min, Ra_max
    σ_r_a, σ_a = σ_r_min, σ_min
    σ_r_b, σ_b = σ_r_max, σ_max

    for iter in 1:max_iter
        # Midpoint
        Ra_mid = (Ra_a + Ra_b) / 2.0

        println("\nIteration $iter:")
        println("  Bracket: [$(Ra_a), $(Ra_b)]")
        println("  Testing Ra = $(Ra_mid)")

        σ_r_mid, σ_mid = growth_rate(Ra_mid)

        # Check convergence
        rel_error = abs(Ra_b - Ra_a) / Ra_mid
        println("  Relative bracket size: $(rel_error)")

        if rel_error < tol
            println("\n" * "="^80)
            println("✓ CONVERGED")
            println("="^80)
            Ra_c = Ra_mid
            ω_c = imag(σ_mid)
            println("Critical Rayleigh number: Ra_c = $(Ra_c)")
            println("Drift frequency: ω_c = $(ω_c)")
            println("Residual growth rate: σ_r = $(σ_r_mid)")
            println("Iterations: $iter")
            println("="^80)
            return Ra_c, ω_c, σ_mid, iter
        end

        # Update bracket
        if σ_r_mid * σ_r_a > 0
            # Same sign as a, replace a
            Ra_a = Ra_mid
            σ_r_a = σ_r_mid
            σ_a = σ_mid
        else
            # Opposite sign, replace b
            Ra_b = Ra_mid
            σ_r_b = σ_r_mid
            σ_b = σ_mid
        end
    end

    @warn "Maximum iterations reached without convergence" max_iter
    Ra_c = (Ra_a + Ra_b) / 2.0
    σ_r_mid, σ_mid = growth_rate(Ra_c)
    ω_c = imag(σ_mid)

    return Ra_c, ω_c, σ_mid, max_iter
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
