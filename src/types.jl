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
struct OnsetProblem{T, BS}
    params::OnsetParams{T, BS}
    function OnsetProblem(params::OnsetParams{T, BS}) where {T, BS}
        validate_onset_params(params)
        new{T, BS}(params)
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
struct BiglobalProblem{T, BS}
    params::OnsetParams{T, BS}
    basic_state::BasicState{T}
    function BiglobalProblem(params::OnsetParams{T, BS}, basic_state::BasicState{T}) where {T, BS}
        validate_onset_params(params)
        validate_basic_state_consistency(basic_state, params)
        new{T, BS}(params, basic_state)
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
struct TriglobalProblem{T, BS}
    params::OnsetParams{T, BS}
    basic_state::BasicState3D{T}
    m_range::UnitRange{Int}
    function TriglobalProblem(params::OnsetParams{T, BS}, basic_state::BasicState3D{T}, m_range::UnitRange{Int}) where {T, BS}
        validate_triglobal_params(params, basic_state, m_range)
        new{T, BS}(params, basic_state, m_range)
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
    params::MHDParams{T}
    basic_state::BS
end

MHDProblem(params::MHDParams{T}) where {T} = MHDProblem{T, Nothing}(params, nothing)

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
    leading_index::Int
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
    StabilityResult{T, P, E}(eigenvalues, eigenvectors, gr, freq, idx, problem, extra)
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
    leading_mode(r::StabilityResult) -> AbstractVector

Return the eigenvector corresponding to the most unstable (largest real part) eigenvalue.
"""
leading_mode(r::StabilityResult) = @view r.eigenvectors[:, r.leading_index]

# --- Problem size estimation (shared helpers) ---

function _count_l_modes(m::Int, lmax::Int, symmetry::Symbol)
    if symmetry == :both
        return lmax - m + 1
    elseif symmetry == :symmetric
        return length(m:2:lmax)
    else  # :antisymmetric
        return length((m+1):2:lmax)
    end
end

# Compute total DOF and memory (GB) for two dense complex matrices (A + B).
function _mem_gb(total_dof::Int)
    return 2 * total_dof^2 * sizeof(ComplexF64) / 1024^3
end

function _hd_total_dof(m::Int, lmax::Int, Nr::Int, symmetry::Symbol)
    n_l = _count_l_modes(m, lmax, symmetry)
    return n_l * (Nr + 1) * 3  # 3 fields: poloidal, toroidal, temperature
end

function _triglobal_total_dof(m_range, lmax::Int, Nr::Int)
    n_l_avg = _count_l_modes(0, lmax, :both)
    dof_per_m = n_l_avg * (Nr + 1) * 3
    return dof_per_m * length(m_range), dof_per_m
end

function _mhd_total_dof(params)
    m, lmax, symm, N = params.m, params.lmax, params.symm, params.N
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
    n_f = params.B0_type == no_field ? 0 : n_pol
    n_g = params.B0_type == no_field ? 0 : n_tor
    total_dof = (n_pol + n_tor + n_f + n_g + n_pol) * n_per_mode
    return total_dof, n_pol, n_tor, n_f, n_g, n_per_mode
end

"""
    estimate_size(p)

Print a human-readable estimate of the matrix size and memory requirement for problem `p`.

Accepts `OnsetProblem`, `BiglobalProblem`, `TriglobalProblem`, or `MHDProblem`.
Warns when the estimated memory exceeds 8 GB.
"""
function estimate_size(p::OnsetProblem)
    params = p.params
    n_l = _count_l_modes(params.m, params.lmax, params.equatorial_symmetry)
    total_dof = _hd_total_dof(params.m, params.lmax, params.Nr, params.equatorial_symmetry)
    mem_gb = _mem_gb(total_dof)

    @printf("OnsetProblem size estimate\n")
    @printf("  l-modes:      %d (m=%d, lmax=%d, %s)\n", n_l, params.m, params.lmax, params.equatorial_symmetry)
    @printf("  DOF per mode: %d (Nr=%d, 3 fields)\n", (params.Nr + 1) * 3, params.Nr)
    @printf("  Total matrix: %d × %d\n", total_dof, total_dof)
    @printf("  Memory (A+B): ~%.1f GB\n", mem_gb)
    mem_gb > 8.0 && @printf("  ⚠ Large problem — consider reducing lmax or Nr\n")
end

function estimate_size(p::BiglobalProblem)
    params = p.params
    n_l = _count_l_modes(params.m, params.lmax, params.equatorial_symmetry)
    total_dof = _hd_total_dof(params.m, params.lmax, params.Nr, params.equatorial_symmetry)
    mem_gb = _mem_gb(total_dof)

    @printf("BiglobalProblem size estimate\n")
    @printf("  l-modes:      %d (m=%d, lmax=%d, %s)\n", n_l, params.m, params.lmax, params.equatorial_symmetry)
    @printf("  DOF per mode: %d (Nr=%d, 3 fields)\n", (params.Nr + 1) * 3, params.Nr)
    @printf("  Total matrix: %d × %d\n", total_dof, total_dof)
    @printf("  Memory (A+B): ~%.1f GB\n", mem_gb)
    mem_gb > 8.0 && @printf("  ⚠ Large problem — consider reducing lmax or Nr\n")
end

function estimate_size(p::TriglobalProblem)
    params = p.params
    total_dof, dof_per_m = _triglobal_total_dof(p.m_range, params.lmax, params.Nr)
    mem_gb = _mem_gb(total_dof)

    @printf("TriglobalProblem size estimate\n")
    @printf("  Coupled modes: m ∈ [%d, %d] (%d modes)\n", first(p.m_range), last(p.m_range), length(p.m_range))
    @printf("  DOF per mode:  ~%d (lmax=%d, Nr=%d, 3 fields)\n", dof_per_m, params.lmax, params.Nr)
    @printf("  Total matrix:  %d × %d\n", total_dof, total_dof)
    @printf("  Memory (A+B):  ~%.1f GB\n", mem_gb)
    mem_gb > 8.0 && @printf("  ⚠ Large problem — consider reducing lmax or m_range\n")
end

function estimate_size(p::MHDProblem)
    total_dof, n_pol, n_tor, _, _, n_per_mode = _mhd_total_dof(p.params)
    mem_gb = _mem_gb(total_dof)
    n_fields = p.params.B0_type == no_field ? 3 : 5

    @printf("MHDProblem size estimate\n")
    @printf("  l-modes:      %d poloidal + %d toroidal (%d fields)\n", n_pol, n_tor, n_fields)
    @printf("  DOF per mode: %d (N=%d)\n", n_per_mode, p.params.N)
    @printf("  Total matrix: %d × %d\n", total_dof, total_dof)
    @printf("  Memory (A+B): ~%.1f GB\n", mem_gb)
    mem_gb > 8.0 && @printf("  ⚠ Large problem — consider reducing lmax or N\n")
end

# --- Makie extension stubs ---

function eigenspectrum end
function plot_meridional end
function plot_radial end
