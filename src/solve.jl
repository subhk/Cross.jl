# ============================================================================
# Unified solve() API for Cross.jl v2.0
# ============================================================================

"""
    solve(problem; nev=6, sigma=nothing)

Solve the eigenvalue problem for the given stability problem.
Returns a `StabilityResult` with eigenvalues, eigenvectors, and analysis-specific data.

Automatically warns if the estimated problem memory exceeds 8 GB.

# Example
```julia
params = OnsetParams(E=1e-3, Pr=1.0, Ra=1e5, χ=0.35, m=4, lmax=20, Nr=32)
result = solve(OnsetProblem(params); nev=6)
result.growth_rate
```
"""
function solve end

# --- Memory warning helper ---

function _warn_if_large(problem, label::String)
    try
        # Redirect estimate_size output and check memory
        buf = IOBuffer()
        # Compute memory inline rather than capturing printed output
        _check_memory(problem, label)
    catch
        # Don't let size estimation failures block solving
    end
end

function _check_memory(p::OnsetProblem, label)
    params = p.params
    n_l = _count_l_modes(params.m, params.lmax, params.equatorial_symmetry)
    n_per_mode = params.Nr + 1
    n_fields = 3
    total_dof = n_l * n_per_mode * n_fields
    mem_bytes = 2 * total_dof^2 * sizeof(ComplexF64)
    mem_gb = mem_bytes / 1024^3
    if mem_gb > 8.0
        @warn "$label: estimated memory ~$(round(mem_gb; digits=1)) GB exceeds 8 GB — consider reducing lmax or Nr"
    end
end

function _check_memory(p::BiglobalProblem, label)
    params = p.params
    n_l = _count_l_modes(params.m, params.lmax, params.equatorial_symmetry)
    n_per_mode = params.Nr + 1
    n_fields = 3
    total_dof = n_l * n_per_mode * n_fields
    mem_bytes = 2 * total_dof^2 * sizeof(ComplexF64)
    mem_gb = mem_bytes / 1024^3
    if mem_gb > 8.0
        @warn "$label: estimated memory ~$(round(mem_gb; digits=1)) GB exceeds 8 GB — consider reducing lmax or Nr"
    end
end

function _check_memory(p::TriglobalProblem, label)
    params = p.params
    m_count = length(p.m_range)
    n_l_avg = _count_l_modes(0, params.lmax, :both)
    n_per_mode = params.Nr + 1
    n_fields = 3
    dof_per_m = n_l_avg * n_per_mode * n_fields
    total_dof = dof_per_m * m_count
    mem_bytes = 2 * total_dof^2 * sizeof(ComplexF64)
    mem_gb = mem_bytes / 1024^3
    if mem_gb > 8.0
        @warn "$label: estimated memory ~$(round(mem_gb; digits=1)) GB exceeds 8 GB — consider reducing lmax or m_range"
    end
end

function _check_memory(p::MHDProblem, label)
    params = p.params
    m = params.m
    lmax = params.lmax
    symm = params.symm
    N = params.N

    # Count l-modes (5 fields for MHD)
    if symm == 1
        n_pol = length(m:2:lmax)
        n_tor = length((m+1):2:lmax)
    elseif symm == -1
        n_pol = length((m+1):2:lmax)
        n_tor = length(m:2:lmax)
    else
        n_pol = length(m:lmax)
        n_tor = length(m:lmax)
    end
    n_per_mode = N + 1
    # 5 fields: u (poloidal vel), v (toroidal vel), f (poloidal mag), g (toroidal mag), h (temperature)
    total_dof = (n_pol + n_tor + n_pol + n_tor + n_pol) * n_per_mode
    mem_bytes = 2 * total_dof^2 * sizeof(ComplexF64)
    mem_gb = mem_bytes / 1024^3
    if mem_gb > 8.0
        @warn "$label: estimated memory ~$(round(mem_gb; digits=1)) GB exceeds 8 GB — consider reducing lmax or N"
    end
end

# ============================================================================
# OnsetProblem solve
# ============================================================================

"""
    solve(problem::OnsetProblem; nev=6, sigma=nothing, tol=1e-10, maxiter=1000, which=:LR)

Solve the onset of convection eigenvalue problem.

Constructs an `OnsetConvectionParams` from the wrapped `OnsetParams`, calls
`solve_onset_problem`, and wraps the result in a `StabilityResult`.

Returns a 4-tuple `(eigenvalues, eigenvectors, operator, info)` are stored in
`result.extra` as `(operator=op, info=info)`.
"""
function solve(problem::OnsetProblem{T};
               nev::Int=6,
               sigma=nothing,
               tol::Float64=1e-10,
               maxiter::Int=1000,
               which::Symbol=:LR) where T

    _warn_if_large(problem, "OnsetProblem")

    params = problem.params
    onset_params = OnsetConvectionParams{T}(
        params.E, params.Pr, params.Ra, params.χ,
        params.m, params.lmax, params.Nr,
        params.mechanical_bc, params.thermal_bc,
        params.equatorial_symmetry
    )

    eigenvalues, eigenvectors, op, info = solve_onset_problem(onset_params;
        nev=nev, tol=tol, maxiter=maxiter, which=which, sigma=sigma)

    # Convert eigenvectors from Vector{Vector{ComplexF64}} to Matrix{Complex{T}}
    evec_matrix = _eigvecs_to_matrix(eigenvalues, eigenvectors, T)

    return StabilityResult(
        Vector{Complex{T}}(eigenvalues),
        evec_matrix,
        problem;
        extra=(operator=op, info=info)
    )
end

# ============================================================================
# BiglobalProblem solve
# ============================================================================

"""
    solve(problem::BiglobalProblem; nev=6, sigma=nothing, tol=1e-10, maxiter=1000, which=:LR, verbose=false)

Solve the biglobal stability eigenvalue problem with axisymmetric mean flow.

Constructs `BiglobalParams` from the wrapped `OnsetParams` and `BasicState`,
calls `solve_biglobal_problem`, and wraps the result in a `StabilityResult`.
"""
function solve(problem::BiglobalProblem{T};
               nev::Int=6,
               sigma=nothing,
               tol::Float64=1e-10,
               maxiter::Int=1000,
               which::Symbol=:LR,
               verbose::Bool=false) where T

    _warn_if_large(problem, "BiglobalProblem")

    params = problem.params
    biglobal_params = BiglobalParams{T}(
        params.E, params.Pr, params.Ra, params.χ,
        params.m, params.lmax, params.Nr,
        problem.basic_state,
        params.mechanical_bc, params.thermal_bc,
        params.equatorial_symmetry
    )

    eigenvalues, eigenvectors, op, info = solve_biglobal_problem(biglobal_params;
        nev=nev, tol=tol, maxiter=maxiter, which=which, sigma=sigma, verbose=verbose)

    evec_matrix = _eigvecs_to_matrix(eigenvalues, eigenvectors, T)

    return StabilityResult(
        Vector{Complex{T}}(eigenvalues),
        evec_matrix,
        problem;
        extra=(operator=op, info=info)
    )
end

# ============================================================================
# TriglobalProblem solve
# ============================================================================

"""
    solve(problem::TriglobalProblem; nev=6, sigma=nothing, verbose=true)

Solve the triglobal stability eigenvalue problem with non-axisymmetric basic state.

Constructs `TriglobalParams` from the wrapped `OnsetParams`, `BasicState3D`,
and `m_range`, calls `solve_triglobal_eigenvalue_problem`, and wraps the result
in a `StabilityResult`.
"""
function solve(problem::TriglobalProblem{T};
               nev::Int=6,
               sigma=nothing,
               verbose::Bool=true) where T

    _warn_if_large(problem, "TriglobalProblem")

    params = problem.params
    triglobal_params = TriglobalParams{T}(
        params.E, params.Pr, params.Ra, params.χ,
        problem.m_range, params.lmax, params.Nr,
        problem.basic_state,
        params.mechanical_bc, params.thermal_bc
    )

    σ_target = sigma === nothing ? 0.0 : sigma
    eigenvalues, eigenvectors = solve_triglobal_eigenvalue_problem(triglobal_params;
        σ_target=σ_target, nev=nev, verbose=verbose)

    # eigenvectors is already a Matrix from the triglobal solver
    evec_matrix = Matrix{Complex{T}}(eigenvectors)

    return StabilityResult(
        Vector{Complex{T}}(eigenvalues),
        evec_matrix,
        problem;
        extra=(coupled_modes=collect(problem.m_range),)
    )
end

# ============================================================================
# MHDProblem solve
# ============================================================================

"""
    solve(problem::MHDProblem; nev=6, sigma=nothing)

Solve the MHD eigenvalue problem.

Constructs an `MHDStabilityOperator` from the MHD parameters, assembles the
matrices via `assemble_mhd_matrices`, solves with `solve_eigenvalue_problem`,
and wraps the result in a `StabilityResult`.
"""
function solve(problem::MHDProblem{T, BS};
               nev::Int=6,
               sigma=nothing,
               tol::Float64=1e-10,
               maxiter::Int=1000,
               which::Symbol=:LR) where {T, BS}

    _warn_if_large(problem, "MHDProblem")

    mhd_params = problem.params
    op = MHDStabilityOperator(mhd_params)
    A, B, interior_dofs, info_assembly = assemble_mhd_matrices(op)

    eigenvalues, eigenvectors, info = solve_eigenvalue_problem(
        A, B; nev=nev, tol=tol, maxiter=maxiter, which=which, sigma=sigma)

    evec_matrix = _eigvecs_to_matrix(eigenvalues, eigenvectors, Float64)

    return StabilityResult(
        Vector{ComplexF64}(eigenvalues),
        evec_matrix,
        problem;
        extra=(operator=op, interior_dofs=interior_dofs, assembly_info=info_assembly)
    )
end

# ============================================================================
# Utility: convert Vector{Vector} to Matrix
# ============================================================================

function _eigvecs_to_matrix(eigenvalues, eigenvectors, ::Type{T}) where T
    if eigenvectors isa AbstractMatrix
        return Matrix{Complex{T}}(eigenvectors)
    elseif eigenvectors isa AbstractVector && !isempty(eigenvectors)
        n = length(eigenvectors[1])
        nev = length(eigenvectors)
        mat = Matrix{Complex{T}}(undef, n, nev)
        for j in 1:nev
            mat[:, j] = eigenvectors[j]
        end
        return mat
    else
        # Fallback: empty matrix
        return Matrix{Complex{T}}(undef, 0, length(eigenvalues))
    end
end

# ============================================================================
# Unified find_critical_Ra API
# ============================================================================

"""
    find_critical_Ra(problem; Ra_guess=1e6, tol=1e-6, kwargs...)

Find the critical Rayleigh number for the given stability problem.
Dispatches to the appropriate solver based on problem type.

# Example
```julia
params = OnsetParams(E=1e-3, Pr=1.0, Ra=1e6, χ=0.35, m=4, lmax=20, Nr=32)
Ra_c = find_critical_Ra(OnsetProblem(params); Ra_guess=1e5)
```
"""
function find_critical_Ra end

function find_critical_Ra(problem::OnsetProblem{T};
                          Ra_guess::T=T(1e6),
                          tol::T=T(1e-6),
                          nev::Int=6,
                          verbose::Bool=false,
                          kwargs...) where T
    p = problem.params
    return find_critical_Ra_onset(;
        E=p.E, Pr=p.Pr, χ=p.χ, m=p.m, lmax=p.lmax, Nr=p.Nr,
        Ra_guess=Ra_guess, tol=tol,
        mechanical_bc=p.mechanical_bc, thermal_bc=p.thermal_bc,
        equatorial_symmetry=p.equatorial_symmetry,
        nev=nev, verbose=verbose, kwargs...)
end

function find_critical_Ra(problem::BiglobalProblem{T};
                          Ra_guess::T=T(1e6),
                          tol::T=T(1e-6),
                          nev::Int=6,
                          verbose::Bool=false,
                          kwargs...) where T
    p = problem.params
    return find_critical_Ra_biglobal(;
        E=p.E, Pr=p.Pr, χ=p.χ, m=p.m, lmax=p.lmax, Nr=p.Nr,
        basic_state=problem.basic_state,
        Ra_guess=Ra_guess, tol=tol,
        mechanical_bc=p.mechanical_bc, thermal_bc=p.thermal_bc,
        nev=nev, verbose=verbose, kwargs...)
end

function find_critical_Ra(problem::TriglobalProblem{T};
                          Ra_min::Real=1e5,
                          Ra_max::Real=1e8,
                          tol::Real=1e-4,
                          max_iter::Int=20,
                          verbose::Bool=true,
                          kwargs...) where T
    p = problem.params
    return find_critical_rayleigh_triglobal(
        p.E, p.Pr, p.χ, problem.m_range, p.lmax, p.Nr,
        problem.basic_state;
        Ra_min=Ra_min, Ra_max=Ra_max, tol=tol, max_iter=max_iter,
        mechanical_bc=p.mechanical_bc, thermal_bc=p.thermal_bc,
        verbose=verbose, kwargs...)
end

function find_critical_Ra(::MHDProblem; kwargs...)
    error("""find_critical_Ra is not supported for MHDProblem.

MHD dynamo problems involve coupled velocity-magnetic field instabilities where
the critical parameter depends on multiple numbers (Ra, Pm, Le) simultaneously.
Use solve(MHDProblem(...); nev=...) directly and inspect the growth rate.""")
end
