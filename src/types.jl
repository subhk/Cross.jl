# ============================================================================
# Shared types for Cross.jl v2.0
# ============================================================================

# --- Abstract base ---
abstract type AbstractStabilityResult{T} end

# --- Problem types ---

struct OnsetProblem{T}
    params::OnsetParams{T, <:Any}
    function OnsetProblem(params::OnsetParams{T, <:Any}) where {T}
        validate_onset_params(params)
        new{T}(params)
    end
end

struct BiglobalProblem{T}
    params::OnsetParams{T, <:Any}
    basic_state::BasicState{T}
    function BiglobalProblem(params::OnsetParams{T, <:Any}, basic_state::BasicState{T}) where {T}
        validate_onset_params(params)
        validate_basic_state_consistency(basic_state, params)
        new{T}(params, basic_state)
    end
end

struct TriglobalProblem{T}
    params::OnsetParams{T, <:Any}
    basic_state::BasicState3D{T}
    m_range::UnitRange{Int}
    function TriglobalProblem(params::OnsetParams{T, <:Any}, basic_state::BasicState3D{T}, m_range::UnitRange{Int}) where {T}
        validate_onset_params(params)
        new{T}(params, basic_state, m_range)
    end
end

struct MHDProblem{T, BS}
    params  # MHDParams — loosely typed to avoid circular dep with CompleteMHD
    basic_state::BS
end

MHDProblem(params) = MHDProblem{Any, Nothing}(params, nothing)

# --- Result type ---

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

growth_rate(r::StabilityResult) = r.growth_rate
frequency(r::StabilityResult) = r.frequency
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

# --- Makie extension stubs ---

function eigenspectrum end
function plot_meridional end
function plot_radial end
