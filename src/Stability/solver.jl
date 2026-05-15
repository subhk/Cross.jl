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

using Logging


struct ShiftInvertLinearMap{LUType,MatType,VecType,SolveVecType}
    lu::LUType
    B::MatType
    temp::VecType
    solve_rhs::SolveVecType
    solve_sol::SolveVecType
end

function (M::ShiftInvertLinearMap)(y, x)
    mul!(M.temp, M.B, x)
    copyto!(M.solve_rhs, M.temp)
    ldiv!(M.solve_sol, M.lu, M.solve_rhs)
    copyto!(y, M.solve_sol)
    return y
end

function _solver_real_eltype(::Type{T}) where {T<:Real}
    return float(T)
end
function _solver_real_eltype(::Type{Complex{T}}) where {T<:Real}
    return T
end

function construct_linear_map(A_shifted::SparseMatrixCSC{T,Int},
                              B::SparseMatrixCSC{T,Int}) where {T<:Complex}
    lu_factor = lu(A_shifted)
    tmp = Vector{T}(undef, size(B, 1))
    solve_tmp = Vector{eltype(lu_factor)}(undef, size(B, 1))
    solve_sol = similar(solve_tmp)
    return LinearMap{T}(ShiftInvertLinearMap(lu_factor, B, tmp, solve_tmp, solve_sol),
                        size(B, 1); ismutating=true)
end

function _is_krylov_convergence_error(err)
    return isdefined(KrylovKit, :ConvergenceError) &&
           err isa getfield(KrylovKit, :ConvergenceError)
end

"""
    solve_eigenvalue_problem(A, B; nev=20, sigma=nothing, which=:LR,
                             selection=:maxreal, tol=1e-10)

Solve the generalized eigenvalue problem A·x = σ·B·x for sparse matrices using KrylovKit
with shift-invert method.

# Arguments
- `A::SparseMatrixCSC`: Operator matrix (physics terms)
- `B::SparseMatrixCSC`: Mass matrix (time derivative weights)
- `nev::Int=20`: Number of eigenvalues to compute
- `sigma::Union{Nothing,Number}=nothing`: Shift target for shift-invert modes
- `which::Symbol=:LR`: Determines automatic shift selection (see below)
- `selection::Symbol=:maxreal`: How to order the returned eigenvalues:
  - `:maxreal`: sort by descending real part (default, best for onset)
  - `:minabs`: sort by ascending magnitude
  - `:closest_real`: sort by ascending |Re(σ)| (best for critical Ra search)
- `tol::Float64=1e-10`: Convergence tolerance
- `maxiter::Int=1000`: Maximum number of iterations
- `krylovdim::Union{Nothing,Int}=nothing`: Krylov subspace dimension
- `verbosity::Int=0`: Verbosity level for KrylovKit

# Returns
- `eigenvalues::Vector{Complex}`: Computed eigenvalues σ = σ_r + iω
- `eigenvectors::Matrix{Complex}`: Corresponding eigenvectors
- `info::Dict`: Information about the solve

# Notes
**SHIFT-INVERT STRATEGY:**
- Uses shift-invert method: solves (A - σ*B)^(-1)*B*x = μ*x where μ = 1/(λ - σ)
- Always uses `:LM` (Largest Magnitude) for the transformed problem
- The `which` parameter determines SHIFT SELECTION:
  - `:LR` → shift σ=10.0 (targets eigenvalues with large positive real part)
  - `:LI` → shift σ=10.0i (targets eigenvalues with large imaginary part)
  - other → shift σ=1.0 (general purpose)
- Results are sorted by `selection` criterion AFTER transformation
- For onset problems: use `which=:LR, selection=:maxreal` (default)
- For critical Ra: use `sigma=0.0, selection=:closest_real`
"""
function solve_eigenvalue_problem(A::SparseMatrixCSC, B::SparseMatrixCSC;
                                 nev::Int=1,
                                 sigma::Union{Nothing,Number}=nothing,
                                 which::Symbol=:LR,  # Determines shift selection (not eigsolve target)
                                 selection::Symbol=:maxreal,
                                 tol::Float64=1e-10,
                                 maxiter::Int=1000,
                                 krylovdim::Union{Nothing,Int}=nothing,
                                 verbosity::Int=0)

    n = size(A, 1)
    size(A) == size(B) || throw(DimensionMismatch(
        "A and B must have same dimensions, got $(size(A)) and $(size(B))"))
    size(A, 1) == size(A, 2) || throw(DimensionMismatch(
        "Matrices must be square, got $(size(A))"))

    @info "Solving eigenvalue problem" solver="KrylovKit shift-invert" size="$n × $n" A_nnz=nnz(A) B_nnz=nnz(B) nev=nev which=which selection=selection

    eigenvalues, eigenvectors, info = _krylov_eigensolve(A, B;
                                                         nev = nev,
                                                         sigma = sigma,
                                                         which = which,
                                                         selection = selection,
                                                         tol = tol,
                                                         maxiter = maxiter,
                                                         krylovdim = krylovdim,
                                                         verbosity = verbosity)

    @info "Eigensolve converged" selected=eigenvalues[1] growth_rate=real(eigenvalues[1]) frequency=imag(eigenvalues[1]) selection=selection
    return eigenvalues, eigenvectors, info
end

function _sort_indices(eigenvalues::AbstractVector{<:Complex}, selection::Symbol)
    select_values = real.(eigenvalues)
    if selection == :maxreal
        return sortperm(select_values, rev=true)
    elseif selection == :minabs
        return sortperm(abs.(eigenvalues))
    elseif selection == :closest_real
        return sortperm(abs.(select_values))
    else
        error("Unknown selection strategy $(selection)")
    end
end

function _krylov_eigensolve(A::SparseMatrixCSC, B::SparseMatrixCSC;
                            nev::Int,
                            sigma::Union{Nothing,Number},
                            which::Symbol,
                            selection::Symbol,
                            tol::Float64,
                            maxiter::Int,
                            krylovdim::Union{Nothing,Int},
                            verbosity::Int,
                            _allow_retry::Bool=true)

    n = size(A, 1)
    T = promote_type(_solver_real_eltype(eltype(A)), _solver_real_eltype(eltype(B)))
    C = Complex{T}

    # Smart shift selection for onset problems (matching linear_stability.jl)
    if sigma === nothing
        if which == :LR
            σ_eff = C(T(10), zero(T))
        elseif which == :LI
            σ_eff = C(zero(T), T(10))
        else
            σ_eff = C(one(T), zero(T))
        end
        @debug "Auto-selected shift" σ=σ_eff which=which
    else
        σ_eff = C(sigma)
        @debug "User-specified shift" σ=σ_eff
    end

    A_complex = SparseMatrixCSC{C, Int}(A)
    B_complex = SparseMatrixCSC{C, Int}(B)
    shift_matrix = A_complex - σ_eff * B_complex

    linmap = construct_linear_map(shift_matrix, B_complex)
    x0 = randn(C, n)

    kdim = krylovdim === nothing ? 300 : krylovdim

    try
        # IMPORTANT: For shift-invert with LinearMap, we must use :LM (Largest Magnitude)
        # to find eigenvalues closest to the shift σ_eff. The 'which' parameter only
        # determines the shift selection, NOT the eigsolve target.
        values, vectors, history = eigsolve(
            linmap, x0, nev, :LM;  # Always use :LM for shift-invert!
            tol = tol,
            maxiter = maxiter,
            krylovdim = kdim,
            verbosity = verbosity
        )

        keep = [abs(val) > eps(T) for val in values]
        any(keep) || error("No finite eigenvalues returned by Krylov solver")
        values = values[keep]
        vectors = vectors[keep]

        eigenvalues = C.(σ_eff .+ inv.(values))
        eigenvectors = hcat(map(v -> C.(v), vectors)...)

        perm = _sort_indices(eigenvalues, selection)
        eigenvalues = eigenvalues[perm]
        eigenvectors = eigenvectors[:, perm]

        info = Dict(
            "solver" => :krylovkit,
            "strategy" => :shift_invert,
            "shift" => σ_eff,
            "krylovdim" => kdim,
            "iterations" => history.numiter,
            "operator_applications" => history.numops,
            "converged" => history.converged,
            "residuals" => history.residual,
            "residual_norms" => history.normres,
            "selected" => eigenvalues[1],
            "selection" => selection
        )

        return eigenvalues, eigenvectors, info
    catch err
        if sigma !== nothing && _allow_retry && _is_krylov_convergence_error(err)
            @warn "Shift-invert Krylov solver failed to converge; retrying without shift" exception=(err, catch_backtrace())
            return _krylov_eigensolve(A, B;
                                      nev = nev,
                                      sigma = nothing,
                                      which = which,
                                      selection = selection,
                                      tol = tol,
                                      maxiter = maxiter,
                                      krylovdim = krylovdim,
                                      verbosity = verbosity,
                                      _allow_retry = false)
        else
            @error "Eigenvalue solver failed" exception=(err, catch_backtrace())
            rethrow()
        end
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
function find_critical_rayleigh(operator_builder::Function, E::TE, χ::Tχ, m::Int;
                               Ra_min=1e4, Ra_max=1e10,
                               tol=1e-6, growth_tol=1e-6,
                               max_iter::Int=50, nev::Int=1) where {TE<:Real, Tχ<:Real}

    T = promote_type(TE, Tχ)
    E = T(E)
    χ = T(χ)
    Ra_min = T(Ra_min)
    Ra_max = T(Ra_max)
    tol = T(tol)
    growth_tol = T(growth_tol)

    @info "Finding critical Rayleigh number" E=E χ=χ m=m bracket="[$Ra_min, $Ra_max]" tol=tol growth_tol=growth_tol

    function _eval_growth_rate(Ra)
        @debug "Testing Ra" Ra=Ra
        A, B = operator_builder(Ra)

        solver_nev = max(nev, 10)
        eigenvalues, _, info = solve_eigenvalue_problem(
            A, B;
            nev = solver_nev,
            sigma = zero(T),
            which = :LR,
            selection = :closest_real
        )

        σ = Complex{T}(eigenvalues[1])
        σ_r = T(real(σ))

        @debug "Growth rate evaluated" Ra=Ra σ_r=σ_r
        return σ_r, σ
    end

    # Cache evaluations to avoid redundant solves when scanning
    known_values = Dict{T, Tuple{T, Complex{T}}}()

    function growth_rate_cached(Ra)
        Ra_key = T(Ra)
        if haskey(known_values, Ra_key)
            return known_values[Ra_key]
        end
        σ_r, σ = _eval_growth_rate(Ra_key)
        known_values[Ra_key] = (σ_r, σ)
        return σ_r, σ
    end

    # Initial bracket check
    @debug "Checking initial bracket..."
    σ_r_min, σ_min = growth_rate_cached(Ra_min)
    if abs(σ_r_min) < growth_tol
        @info "Lower bracket already satisfies growth tolerance" Ra=Ra_min
        return Ra_min, imag(σ_min), σ_min, 0
    end

    σ_r_max, σ_max = growth_rate_cached(Ra_max)
    if abs(σ_r_max) < growth_tol
        @info "Upper bracket already satisfies growth tolerance" Ra=Ra_max
        return Ra_max, imag(σ_max), σ_max, 0
    end

    if σ_r_min * σ_r_max > 0
        @warn "Initial bracket may not contain critical Ra" σ_r_min σ_r_max
        max_bracket_expansions = 12
        expansion_iter = 0
        while σ_r_min * σ_r_max > 0 && expansion_iter < max_bracket_expansions
            expansion_iter += 1
            if σ_r_min > 0 && σ_r_max > 0
                Ra_min /= T(2)
                @debug "Bracket expansion: lowering Ra_min" iter=expansion_iter Ra_min=Ra_min
                if Ra_min <= 0
                    error("Lower Rayleigh bound reached non-positive value while trying to bracket root.")
                end
                σ_r_min, σ_min = growth_rate_cached(Ra_min)
            elseif σ_r_min < 0 && σ_r_max < 0
                Ra_max *= T(2)
                @debug "Bracket expansion: raising Ra_max" iter=expansion_iter Ra_max=Ra_max
                σ_r_max, σ_max = growth_rate_cached(Ra_max)
            else
                break
            end
        end
        if σ_r_min * σ_r_max > 0
            @debug "Expansion exhausted, performing logarithmic scan..."
            min_scan = max(Ra_min, T(10)) / T(10)
            max_scan = Ra_max * T(10)
            if min_scan <= 0
                min_scan = tol
            end
            scan_points = 30
            scan_values = exp10.(LinRange(log10(min_scan), log10(max_scan), scan_points))
            scan_values = sort(unique(vcat(Ra_min, Ra_max, scan_values)))

            bracket_found = false
            last_ra = nothing
            lastσ_r = zero(T)
            lastσ = zero(Complex{T})

            for Ra in scan_values
                σ_r, σ = growth_rate_cached(Ra)
                if last_ra !== nothing && σ_r * lastσ_r <= 0
                    @debug "Found sign change during scan" Ra_low=last_ra Ra_high=Ra
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
    @debug "Starting Brent search..."
    Ra_a, Ra_b, Ra_c = Ra_min, Ra_max, Ra_min
    σ_r_a, σ_r_b, σ_r_c = σ_r_min, σ_r_max, σ_r_min
    σ_a, σ_b, σ_c = σ_min, σ_max, σ_min

    if σ_r_a * σ_r_b >= 0
        error("Brent search requires opposite signs at the bracket endpoints.")
    end

    abs_tol = tol * max(abs(Ra_a), abs(Ra_b), one(T))
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

        tol_act = T(2) * eps(abs(Ra_b)) + abs_tol
        half_width = T(0.5) * (Ra_c - Ra_b)

        @debug "Brent iteration" iter=iter bracket="[$(Ra_a), $(Ra_c)]" Ra=Ra_b σ_r_a=σ_r_a σ_r_b=σ_r_b σ_r_c=σ_r_c

        if abs(σ_r_b) < growth_tol
            Ra_c_final = Ra_b
            ω_c = imag(σ_b)
            @info "Critical Ra converged" Ra_c=Ra_c_final ω_c=ω_c σ_r=σ_r_b iterations=iter
            return Ra_c_final, ω_c, σ_b, iter
        elseif abs(half_width) <= tol_act
            @debug "Bracket tolerance met but growth rate exceeds growth_tol" abs_σ_r=abs(σ_r_b) growth_tol=growth_tol
        end

        if abs(e) < tol_act || abs(σ_r_a) <= abs(σ_r_b)
            d = half_width
            e = half_width
            @debug "Using bisection step"
        else
            s = σ_r_b / σ_r_a
            if Ra_a == Ra_c
                # Secant method
                p = T(2) * half_width * s
                q = one(T) - s
            else
                q = σ_r_a / σ_r_c
                r = σ_r_b / σ_r_c
                p = s * (T(2) * half_width * q * (q - r) - (Ra_b - Ra_a) * (r - one(T)))
                q = (q - one(T)) * (r - one(T)) * (s - one(T))
            end

            if p > zero(T)
                q = -q
            else
                p = -p
            end

            if (T(2) * p < T(3) * half_width * q - abs(tol_act * q)) &&
                    (p < abs(T(0.5) * e * q))
                e = d
                d = p / q
                @debug "Using inverse interpolation step"
            else
                d = half_width
                e = half_width
                @debug "Interpolation rejected, falling back to bisection"
            end
        end

        Ra_a = Ra_b
        σ_r_a = σ_r_b
        σ_a = σ_b

        if abs(d) > tol_act
            Ra_b += d
        else
            Ra_b += half_width >= 0 ? tol_act : -tol_act
        end

        σ_r_b, σ_b = growth_rate_cached(Ra_b)

        if abs(σ_r_b) < growth_tol
            ω_c = imag(σ_b)
            @info "Critical Ra converged" Ra_c=Ra_b ω_c=ω_c σ_r=σ_r_b iterations=iter
            return Ra_b, ω_c, σ_b, iter
        end
    end

    @warn "Maximum iterations reached without convergence (Brent)"
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
                               E::TE, χ::Tχ, Pr::TP,
                               m_range::AbstractVector{Int};
                               kwargs...) where {TE<:Real, Tχ<:Real, TP<:Real}

    T = promote_type(TE, Tχ, TP)
    E = T(E)
    χ = T(χ)
    Pr = T(Pr)

    @info "Scanning for onset parameters" E=E χ=χ Pr=Pr m_range=m_range

    SuccessResult = NamedTuple{(:Ra_c, :ω_c, :σ_c, :iters),
        Tuple{T, T, Complex{T}, Int}}
    ErrorResult = NamedTuple{(:error,), Tuple{Exception}}
    Result = Union{SuccessResult, ErrorResult}
    results = Dict{Int, Result}()
    Ra_c_min = T(Inf)
    m_c = 0
    ω_c_best = zero(T)

    for m in m_range
        @info "Testing mode" m=m

        try
            operator_builder = operator_builder_factory(E, χ, Pr, m)

            Ra_c, ω_c, σ_c, iters = find_critical_rayleigh(
                operator_builder, E, χ, m; kwargs...
            )

            results[m] = SuccessResult((T(Ra_c), T(ω_c), Complex{T}(σ_c), iters))

            if Ra_c < Ra_c_min
                Ra_c_min = T(Ra_c)
                m_c = m
                ω_c_best = T(ω_c)
            end

            @info "Mode result" m=m Ra_c=Ra_c ω_c=ω_c

        catch err
            @warn "Failed for mode" m=m exception=err
            results[m] = ErrorResult((err,))
        end
    end

    @info "Onset parameters found" m_c=m_c Ra_c=Ra_c_min ω_c=ω_c_best

    return Ra_c_min, m_c, ω_c_best, results
end
