# ============================================================================
# Pretty-printing for Cross.jl public types
# ============================================================================

import Base: show

# --- OnsetParams ---
function show(io::IO, ::MIME"text/plain", p::OnsetParams{T}) where T
    println(io, "OnsetParams{$T}")
    println(io, "  E  = $(p.E)    Pr = $(p.Pr)    Ra = $(p.Ra)    χ = $(p.χ)")
    println(io, "  m  = $(p.m)         lmax = $(p.lmax)   Nr = $(p.Nr)")
    println(io, "  BCs: $(p.mechanical_bc) | $(p.thermal_bc)")
    print(io,   "  Symmetry: $(p.equatorial_symmetry)")
end

# --- BasicState ---
function show(io::IO, ::MIME"text/plain", bs::BasicState{T}) where T
    active_theta = sort(collect(keys(bs.theta_coeffs)))
    active_uphi = sort(collect(keys(bs.uphi_coeffs)))
    r_min = isempty(bs.r) ? "?" : @sprintf("%.3f", minimum(bs.r))
    r_max = isempty(bs.r) ? "?" : @sprintf("%.3f", maximum(bs.r))

    println(io, "BasicState{$T}")
    println(io, "  Nr = $(bs.Nr)    lmax_bs = $(bs.lmax_bs)")
    theta_str = isempty(active_theta) ? "none" : join(["ℓ=$l" for l in active_theta], ",")
    uphi_str = isempty(active_uphi) ? "none" : join(["ℓ=$l" for l in active_uphi], ",")
    println(io, "  Active modes: θ̄($theta_str)  ūφ($uphi_str)")
    print(io,   "  Domain: r ∈ [$r_min, $r_max]")
end

# --- BasicState3D ---
function show(io::IO, ::MIME"text/plain", bs::BasicState3D{T}) where T
    n_modes = length(bs.theta_coeffs)
    println(io, "BasicState3D{$T}")
    println(io, "  Nr = $(bs.Nr)    lmax_bs = $(bs.lmax_bs)    mmax_bs = $(bs.mmax_bs)")
    print(io,   "  Active (ℓ,m) modes: $n_modes")
end

# --- StabilityResult ---
function show(io::IO, ::MIME"text/plain", r::StabilityResult{T}) where T
    nev = length(r.eigenvalues)
    println(io, "StabilityResult ($nev eigenvalues)")
    println(io, "  Growth rate: $(r.growth_rate) + $(r.frequency)im")
    pname = _problem_name(r.problem)
    print(io,   "  Problem: $pname")
end

_problem_name(p::OnsetProblem) = "OnsetProblem (E=$(p.params.E), Ra=$(p.params.Ra))"
_problem_name(p::BiglobalProblem) = "BiglobalProblem (E=$(p.params.E), Ra=$(p.params.Ra))"
_problem_name(p::TriglobalProblem) = "TriglobalProblem (E=$(p.params.E), m=$(p.m_range))"
function _problem_name(p::MHDProblem)
    try
        mp = p.params
        return "MHDProblem (E=$(mp.E), Ra=$(mp.Ra), Pm=$(mp.Pm), Le=$(mp.Le), m=$(mp.m))"
    catch
        return "MHDProblem"
    end
end
_problem_name(::Any) = "Unknown"

# --- Problem types ---
function show(io::IO, ::MIME"text/plain", p::OnsetProblem{T}) where T
    println(io, "OnsetProblem{$T}")
    print(io,   "  E=$(p.params.E)  Ra=$(p.params.Ra)  m=$(p.params.m)  lmax=$(p.params.lmax)  Nr=$(p.params.Nr)")
end

function show(io::IO, ::MIME"text/plain", p::BiglobalProblem{T}) where T
    println(io, "BiglobalProblem{$T}")
    println(io, "  E=$(p.params.E)  Ra=$(p.params.Ra)  m=$(p.params.m)  lmax=$(p.params.lmax)  Nr=$(p.params.Nr)")
    print(io,   "  BasicState: lmax_bs=$(p.basic_state.lmax_bs)")
end

function show(io::IO, ::MIME"text/plain", p::TriglobalProblem{T}) where T
    println(io, "TriglobalProblem{$T}")
    println(io, "  E=$(p.params.E)  Ra=$(p.params.Ra)  lmax=$(p.params.lmax)  Nr=$(p.params.Nr)")
    print(io,   "  m_range=$(p.m_range) ($(length(p.m_range)) coupled modes)")
end

# --- BiglobalParams ---
function show(io::IO, ::MIME"text/plain", p::BiglobalParams{T}) where T
    println(io, "BiglobalParams{$T}")
    println(io, "  E  = $(p.E)    Pr = $(p.Pr)    Ra = $(p.Ra)    χ = $(p.χ)")
    println(io, "  m  = $(p.m)         lmax = $(p.lmax)   Nr = $(p.Nr)")
    println(io, "  BCs: $(p.mechanical_bc) | $(p.thermal_bc)")
    print(io,   "  BasicState: lmax_bs=$(p.basic_state.lmax_bs)")
end

# --- TriglobalParams ---
function show(io::IO, ::MIME"text/plain", p::TriglobalParams{T}) where T
    println(io, "TriglobalParams{$T}")
    println(io, "  E  = $(p.E)    Pr = $(p.Pr)    Ra = $(p.Ra)    χ = $(p.χ)")
    println(io, "  m_range = $(p.m_range)    lmax = $(p.lmax)   Nr = $(p.Nr)")
    println(io, "  BCs: $(p.mechanical_bc) | $(p.thermal_bc)")
    print(io,   "  BasicState3D: lmax_bs=$(p.basic_state_3d.lmax_bs), mmax_bs=$(p.basic_state_3d.mmax_bs)")
end

# --- MHDParams ---
function show(io::IO, ::MIME"text/plain", p::MHDParams{T}) where T
    println(io, "MHDParams{$T}")
    println(io, "  E  = $(p.E)    Pr = $(p.Pr)    Pm = $(p.Pm)    Ra = $(p.Ra)")
    println(io, "  Le = $(p.Le)   ricb = $(p.ricb)    m = $(p.m)    lmax = $(p.lmax)")
    println(io, "  N  = $(p.N)    symm = $(p.symm)")
    println(io, "  B0: $(p.B0_type) (amplitude=$(p.B0_amplitude))")
    println(io, "  Mechanical BCs: inner=$(p.bci), outer=$(p.bco)")
    println(io, "  Thermal BCs:    inner=$(p.bci_thermal), outer=$(p.bco_thermal)")
    println(io, "  Magnetic BCs:   inner=$(p.bci_magnetic), outer=$(p.bco_magnetic)")
    print(io,   "  Heating: $(p.heating)")
end

# --- MHDProblem ---
function show(io::IO, ::MIME"text/plain", p::MHDProblem{T, BS}) where {T, BS}
    println(io, "MHDProblem{$T, $BS}")
    try
        mp = p.params
        println(io, "  E=$(mp.E)  Ra=$(mp.Ra)  Pm=$(mp.Pm)  Le=$(mp.Le)  m=$(mp.m)")
        print(io,   "  lmax=$(mp.lmax)  N=$(mp.N)  B0=$(mp.B0_type)")
    catch
        print(io,   "  (params type: $(typeof(p.params)))")
    end
end
