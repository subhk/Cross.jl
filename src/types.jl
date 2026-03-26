# ============================================================================
# Shared types for Cross.jl v2.0
# ============================================================================

# --- Abstract base ---
abstract type AbstractStabilityResult{T} end

# --- Problem types ---

"""
    OnsetProblem{T}

Standard linear onset problem for rotating spherical shell convection.

Wraps an `OnsetParams` and validates parameters on construction.
Use `estimate_size` to preview memory requirements before solving.

# Fields
- `params::OnsetParams{T}` — problem parameters

# Example
```julia
p = OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=30, Nr=64))
```
"""
struct OnsetProblem{T}
    params::OnsetParams{T, <:Any}
    function OnsetProblem(params::OnsetParams{T, <:Any}) where {T}
        validate_onset_params(params)
        new{T}(params)
    end
end

"""
    BiglobalProblem{T}

Biglobal instability problem: onset on an axisymmetric basic state.

Combines `OnsetParams` with a `BasicState` (axisymmetric) and validates
consistency between them on construction.

# Fields
- `params::OnsetParams{T}` — problem parameters
- `basic_state::BasicState{T}` — axisymmetric background state

# Example
```julia
p = BiglobalProblem(params, basic_state)
```
"""
struct BiglobalProblem{T}
    params::OnsetParams{T, <:Any}
    basic_state::BasicState{T}
    function BiglobalProblem(params::OnsetParams{T, <:Any}, basic_state::BasicState{T}) where {T}
        validate_onset_params(params)
        validate_basic_state_consistency(basic_state, params)
        new{T}(params, basic_state)
    end
end

"""
    TriglobalProblem{T}

Triglobal instability problem: onset on a fully 3D (non-axisymmetric) basic state.

Couples multiple azimuthal wavenumbers `m` simultaneously via a `BasicState3D`.

# Fields
- `params::OnsetParams{T}` — problem parameters
- `basic_state::BasicState3D{T}` — 3D background state
- `m_range::UnitRange{Int}` — range of coupled azimuthal wavenumbers

# Example
```julia
p = TriglobalProblem(params, basic_state_3d, 0:4)
```
"""
struct TriglobalProblem{T}
    params::OnsetParams{T, <:Any}
    basic_state::BasicState3D{T}
    m_range::UnitRange{Int}
    function TriglobalProblem(params::OnsetParams{T, <:Any}, basic_state::BasicState3D{T}, m_range::UnitRange{Int}) where {T}
        validate_triglobal_params(params, basic_state, m_range)
        new{T}(params, basic_state, m_range)
    end
end

"""
    MHDProblem{T, BS}

Magnetohydrodynamic instability problem.

Loosely typed to avoid circular dependencies with the `CompleteMHD` module.
`basic_state` may be `nothing` for problems without an explicit background field.

# Fields
- `params` — MHD parameters (e.g., `MHDParams`)
- `basic_state::BS` — background state, or `nothing`
"""
struct MHDProblem{T, BS}
    params  # MHDParams — loosely typed to avoid circular dep with CompleteMHD
    basic_state::BS
end

MHDProblem(params) = MHDProblem{Any, Nothing}(params, nothing)

# --- Result type ---

"""
    StabilityResult{T<:Real, P, E}

Container for the output of a linear stability solve.

Stores all eigenvalues and eigenvectors and pre-extracts the leading (most
unstable) mode's growth rate and oscillation frequency.

# Fields
- `eigenvalues::Vector{Complex{T}}` — all computed eigenvalues
- `eigenvectors::Matrix{Complex{T}}` — corresponding eigenvectors (columns)
- `growth_rate::T` — real part of the leading eigenvalue
- `frequency::T` — imaginary part of the leading eigenvalue
- `problem::P` — the problem that was solved
- `extra::E` — optional solver metadata (default `(;)`)

# Example
```julia
result = solve(problem)
println(growth_rate(result))   # most unstable growth rate
println(frequency(result))     # oscillation frequency
mode = leading_mode(result)    # eigenvector of the leading mode
```
"""
struct StabilityResult{T<:Real, P, E} <: AbstractStabilityResult{T}
    eigenvalues::Vector{Complex{T}}
    eigenvectors::Matrix{Complex{T}}
    growth_rate::T
    frequency::T
    problem::P
    extra::E
end

function StabilityResult(
    eigenvalues::Vector{Complex{T}},
    eigenvectors::Matrix{Complex{T}},
    problem::P;
    extra::E=(;)
) where {T<:Real, P, E}
    idx = argmax(real.(eigenvalues))
    gr = real(eigenvalues[idx])
    freq = imag(eigenvalues[idx])
    StabilityResult{T, P, E}(eigenvalues, eigenvectors, gr, freq, problem, extra)
end

# --- Convenience accessors ---

"""
    growth_rate(r::StabilityResult) -> Real

Return the growth rate (real part of the leading eigenvalue) from a stability result.
"""
growth_rate(r::StabilityResult) = r.growth_rate

"""
    frequency(r::StabilityResult) -> Real

Return the oscillation frequency (imaginary part of the leading eigenvalue) from a stability result.
"""
frequency(r::StabilityResult) = r.frequency

"""
    leading_mode(r::StabilityResult) -> Vector

Return the eigenvector corresponding to the most unstable (largest real part) eigenvalue.
"""
leading_mode(r::StabilityResult) = r.eigenvectors[:, argmax(real.(r.eigenvalues))]

# --- Problem size estimation ---

function _count_l_modes(m::Int, lmax::Int, symmetry::Symbol)
    if symmetry == :both
        return lmax - m + 1
    elseif symmetry == :symmetric
        return length(m:2:lmax)
    else  # :antisymmetric
        return length((m+1):2:lmax)
    end
end

"""
    estimate_size(p)

Print a human-readable estimate of the matrix size and memory requirement for problem `p`.

Accepts `OnsetProblem`, `BiglobalProblem`, or `TriglobalProblem`. Warns when the
estimated memory exceeds 8 GB.
"""
function estimate_size(p::OnsetProblem)
    params = p.params
    n_l = _count_l_modes(params.m, params.lmax, params.equatorial_symmetry)
    n_per_mode = params.Nr + 1
    n_fields = 3
    total_dof = n_l * n_per_mode * n_fields
    mem_bytes = 2 * total_dof^2 * sizeof(ComplexF64)
    mem_gb = mem_bytes / 1024^3

    @printf("OnsetProblem size estimate\n")
    @printf("  l-modes:      %d (m=%d, lmax=%d, %s)\n", n_l, params.m, params.lmax, params.equatorial_symmetry)
    @printf("  DOF per mode: %d (Nr=%d, 3 fields)\n", n_per_mode * n_fields, params.Nr)
    @printf("  Total matrix: %d × %d\n", total_dof, total_dof)
    @printf("  Memory (A+B): ~%.1f GB\n", mem_gb)
    if mem_gb > 8.0
        @printf("  ⚠ Large problem — consider reducing lmax or Nr\n")
    end
end

function estimate_size(p::BiglobalProblem)
    params = p.params
    n_l = _count_l_modes(params.m, params.lmax, params.equatorial_symmetry)
    n_per_mode = params.Nr + 1
    n_fields = 3
    total_dof = n_l * n_per_mode * n_fields
    mem_bytes = 2 * total_dof^2 * sizeof(ComplexF64)
    mem_gb = mem_bytes / 1024^3

    @printf("BiglobalProblem size estimate\n")
    @printf("  l-modes:      %d (m=%d, lmax=%d, %s)\n", n_l, params.m, params.lmax, params.equatorial_symmetry)
    @printf("  DOF per mode: %d (Nr=%d, 3 fields)\n", n_per_mode * n_fields, params.Nr)
    @printf("  Total matrix: %d × %d\n", total_dof, total_dof)
    @printf("  Memory (A+B): ~%.1f GB\n", mem_gb)
    if mem_gb > 8.0
        @printf("  ⚠ Large problem — consider reducing lmax or Nr\n")
    end
end

function estimate_size(p::TriglobalProblem)
    params = p.params
    m_count = length(p.m_range)
    n_l_avg = _count_l_modes(0, params.lmax, :both)
    n_per_mode = params.Nr + 1
    n_fields = 3
    dof_per_m = n_l_avg * n_per_mode * n_fields
    total_dof = dof_per_m * m_count
    mem_bytes = 2 * total_dof^2 * sizeof(ComplexF64)
    mem_gb = mem_bytes / 1024^3

    @printf("TriglobalProblem size estimate\n")
    @printf("  Coupled modes: m ∈ [%d, %d] (%d modes)\n", first(p.m_range), last(p.m_range), m_count)
    @printf("  DOF per mode:  ~%d (lmax=%d, Nr=%d, 3 fields)\n", dof_per_m, params.lmax, params.Nr)
    @printf("  Total matrix:  %d × %d\n", total_dof, total_dof)
    @printf("  Memory (A+B):  ~%.1f GB\n", mem_gb)
    if mem_gb > 8.0
        @printf("  ⚠ Large problem — consider reducing lmax or m_range\n")
    end
end

function estimate_size(p::MHDProblem)
    params = p.params
    m = params.m
    lmax = params.lmax
    symm = params.symm
    N = params.N

    # MHD has 5 fields: poloidal vel (u), toroidal vel (v),
    # poloidal mag (f), toroidal mag (g), temperature (h)
    if symm == 1
        n_pol = length(m:2:lmax)
        n_tor = length((m+1):2:lmax)
    elseif symm == -1
        n_pol = length((m+1):2:lmax)
        n_tor = length(m:2:lmax)
    else  # symm == 0
        n_pol = length(m:lmax)
        n_tor = length(m:lmax)
    end

    n_per_mode = N + 1

    # Magnetic field modes depend on B0_type and parity
    if params.B0_type == no_field
        n_f = 0
        n_g = 0
    else
        # For non-zero background field, magnetic modes mirror velocity modes
        # with possible parity swap (depends on B0 symmetry)
        n_f = n_pol
        n_g = n_tor
    end

    n_u = n_pol * n_per_mode
    n_v = n_tor * n_per_mode
    n_mag_f = n_f * n_per_mode
    n_mag_g = n_g * n_per_mode
    n_h = n_pol * n_per_mode  # temperature has same parity as poloidal velocity
    total_dof = n_u + n_v + n_mag_f + n_mag_g + n_h

    mem_bytes = 2 * total_dof^2 * sizeof(ComplexF64)
    mem_gb = mem_bytes / 1024^3

    n_fields = params.B0_type == no_field ? 3 : 5

    @printf("MHDProblem size estimate\n")
    @printf("  l-modes:      %d poloidal + %d toroidal (%d fields)\n", n_pol, n_tor, n_fields)
    @printf("  DOF per mode: %d (N=%d)\n", n_per_mode, N)
    @printf("  Total matrix: %d × %d\n", total_dof, total_dof)
    @printf("  Memory (A+B): ~%.1f GB\n", mem_gb)
    if mem_gb > 8.0
        @printf("  ⚠ Large problem — consider reducing lmax or N\n")
    end
end

# --- Makie extension stubs ---

function eigenspectrum end
function plot_meridional end
function plot_radial end
