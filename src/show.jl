# ============================================================================
# Pretty-printing for Cross.jl public types
# ============================================================================

import Base: show

"""Print one Oceananigans-style tree row."""
function _tree_row(io::IO, label::AbstractString, value; last::Bool=false)
    branch = last ? "└── " : "├── "
    print(io, branch, label, ": ", value)
    last || println(io)
    return nothing
end

"""Format a radial domain for display."""
function _domain_summary(r)
    isempty(r) && return "unknown"
    r_min = @sprintf("%.3f", minimum(r))
    r_max = @sprintf("%.3f", maximum(r))
    return "[$r_min, $r_max]"
end

"""Join active harmonic degrees for compact display."""
function _degree_summary(keys_iter)
    degrees = sort(collect(keys_iter))
    isempty(degrees) && return "none"
    return join(("ℓ=$ℓ" for ℓ in degrees), ", ")
end

# --- OnsetParams ---
"""Pretty-print hydrodynamic onset parameters in a compact REPL summary."""
function show(io::IO, ::MIME"text/plain", p::OnsetParams{T}) where T
    println(io, "OnsetParams{$T}")
    _tree_row(io, "dynamics", "E=$(p.E), Pr=$(p.Pr), Ra=$(p.Ra)")
    _tree_row(io, "geometry", "χ=$(p.χ), ri=$(p.ri), ro=$(p.ro), L=$(p.L)")
    _tree_row(io, "resolution", "m=$(p.m), lmax=$(p.lmax), Nr=$(p.Nr)")
    _tree_row(io, "boundary conditions", "mechanical=$(p.mechanical_bc), thermal=$(p.thermal_bc)")
    _tree_row(io, "equatorial symmetry", p.equatorial_symmetry; last=true)
end

# --- BasicState ---
"""Pretty-print active axisymmetric temperature and zonal-flow modes."""
function show(io::IO, ::MIME"text/plain", bs::BasicState{T}) where T
    println(io, "BasicState{$T}")
    _tree_row(io, "resolution", "lmax_bs=$(bs.lmax_bs), Nr=$(bs.Nr)")
    _tree_row(io, "temperature modes", _degree_summary(keys(bs.theta_coeffs)))
    _tree_row(io, "zonal-flow modes", _degree_summary(keys(bs.uphi_coeffs)))
    _tree_row(io, "radial domain", _domain_summary(bs.r); last=true)
end

# --- BasicState3D ---
"""Pretty-print the dimensions and active-mode count for a 3D basic state."""
function show(io::IO, ::MIME"text/plain", bs::BasicState3D{T}) where T
    n_modes = length(bs.theta_coeffs)
    println(io, "BasicState3D{$T}")
    _tree_row(io, "resolution", "lmax_bs=$(bs.lmax_bs), mmax_bs=$(bs.mmax_bs), Nr=$(bs.Nr)")
    _tree_row(io, "active temperature modes", n_modes)
    _tree_row(io, "radial domain", _domain_summary(bs.r); last=true)
end

# --- StabilityResult ---
"""Pretty-print the leading eigenvalue summary and source problem."""
function show(io::IO, ::MIME"text/plain", r::StabilityResult{T}) where T
    nev = length(r.eigenvalues)
    leading_λ = r.eigenvalues[r.leading_index]
    println(io, "StabilityResult{$T} with $nev eigenvalues")
    _tree_row(io, "leading eigenvalue", leading_λ)
    _tree_row(io, "growth rate", r.growth_rate)
    _tree_row(io, "frequency", r.frequency)
    _tree_row(io, "problem", _problem_name(r.problem); last=true)
end

"""Build the short problem label embedded in `StabilityResult` display output."""
_problem_name(p::OnsetProblem) = "OnsetProblem (E=$(p.params.E), Ra=$(p.params.Ra))"

"""Build the short biglobal problem label embedded in `StabilityResult` display output."""
_problem_name(p::BiglobalProblem) = "BiglobalProblem (E=$(p.params.E), Ra=$(p.params.Ra))"

"""Build the short triglobal problem label embedded in `StabilityResult` display output."""
_problem_name(p::TriglobalProblem) = "TriglobalProblem (E=$(p.params.E), m=$(p.m_range))"

"""Build the MHD problem label while tolerating incomplete custom params."""
function _problem_name(p::MHDProblem)
    try
        mp = p.params
        return "MHDProblem (E=$(mp.E), Ra=$(mp.Ra), Pm=$(mp.Pm), Le=$(mp.Le), m=$(mp.m))"
    catch
        return "MHDProblem"
    end
end

"""Fallback problem label for unknown result wrappers."""
_problem_name(::Any) = "Unknown"

# --- Problem types ---
"""Pretty-print the defining resolution and physics for an onset wrapper."""
function show(io::IO, ::MIME"text/plain", p::OnsetProblem{T}) where T
    println(io, "OnsetProblem{$T}")
    _tree_row(io, "parameters", "E=$(p.params.E), Ra=$(p.params.Ra), Pr=$(p.params.Pr), χ=$(p.params.χ)")
    _tree_row(io, "resolution", "m=$(p.params.m), lmax=$(p.params.lmax), Nr=$(p.params.Nr)")
    _tree_row(io, "boundary conditions", "mechanical=$(p.params.mechanical_bc), thermal=$(p.params.thermal_bc)"; last=true)
end

"""Pretty-print the defining resolution and attached axisymmetric basic state."""
function show(io::IO, ::MIME"text/plain", p::BiglobalProblem{T}) where T
    println(io, "BiglobalProblem{$T}")
    _tree_row(io, "parameters", "E=$(p.params.E), Ra=$(p.params.Ra), Pr=$(p.params.Pr), χ=$(p.params.χ)")
    _tree_row(io, "resolution", "m=$(p.params.m), lmax=$(p.params.lmax), Nr=$(p.params.Nr)")
    _tree_row(io, "basic state", "BasicState with lmax_bs=$(p.basic_state.lmax_bs)"; last=true)
end

"""Pretty-print the coupled-mode range and resolution for a triglobal wrapper."""
function show(io::IO, ::MIME"text/plain", p::TriglobalProblem{T}) where T
    println(io, "TriglobalProblem{$T}")
    _tree_row(io, "parameters", "E=$(p.params.E), Ra=$(p.params.Ra), Pr=$(p.params.Pr), χ=$(p.params.χ)")
    _tree_row(io, "resolution", "lmax=$(p.params.lmax), Nr=$(p.params.Nr)")
    _tree_row(io, "coupled modes", "$(p.m_range) ($(length(p.m_range)) modes)"; last=true)
end

# --- BiglobalParams ---
"""Pretty-print biglobal solver parameters and basic-state resolution."""
function show(io::IO, ::MIME"text/plain", p::BiglobalParams{T}) where T
    println(io, "BiglobalParams{$T}")
    _tree_row(io, "dynamics", "E=$(p.E), Pr=$(p.Pr), Ra=$(p.Ra)")
    _tree_row(io, "geometry", "χ=$(p.χ)")
    _tree_row(io, "resolution", "m=$(p.m), lmax=$(p.lmax), Nr=$(p.Nr)")
    _tree_row(io, "basic state", "lmax_bs=$(p.basic_state.lmax_bs)"; last=true)
end

# --- TriglobalParams ---
"""Pretty-print triglobal solver parameters, symmetry, and 3D basic-state resolution."""
function show(io::IO, ::MIME"text/plain", p::TriglobalParams{T}) where T
    println(io, "TriglobalParams{$T}")
    _tree_row(io, "dynamics", "E=$(p.E), Pr=$(p.Pr), Ra=$(p.Ra)")
    _tree_row(io, "geometry", "χ=$(p.χ)")
    _tree_row(io, "resolution", "m_range=$(p.m_range), lmax=$(p.lmax), Nr=$(p.Nr)")
    _tree_row(io, "equatorial symmetry", p.equatorial_symmetry)
    _tree_row(io, "basic state", "BasicState3D with lmax_bs=$(p.basic_state_3d.lmax_bs), mmax_bs=$(p.basic_state_3d.mmax_bs)"; last=true)
end

# --- MHDParams ---
"""Pretty-print MHD solver parameters, boundary conditions, and background field."""
function show(io::IO, ::MIME"text/plain", p::MHDParams{T}) where T
    println(io, "MHDParams{$T}")
    _tree_row(io, "dynamics", "E=$(p.E), Pr=$(p.Pr), Pm=$(p.Pm), Ra=$(p.Ra), Le=$(p.Le)")
    _tree_row(io, "geometry", "ricb=$(p.ricb)")
    _tree_row(io, "resolution", "m=$(p.m), lmax=$(p.lmax), N=$(p.N), symm=$(p.symm)")
    _tree_row(io, "background field", "$(p.B0_type) (amplitude=$(p.B0_amplitude))")
    _tree_row(io, "mechanical BCs", "inner=$(p.bci), outer=$(p.bco)")
    _tree_row(io, "thermal BCs", "inner=$(p.bci_thermal), outer=$(p.bco_thermal)")
    _tree_row(io, "magnetic BCs", "inner=$(p.bci_magnetic), outer=$(p.bco_magnetic)")
    _tree_row(io, "heating", p.heating; last=true)
end

# --- MHDProblem ---
"""Pretty-print an MHD problem wrapper while tolerating custom parameter objects."""
function show(io::IO, ::MIME"text/plain", p::MHDProblem{T, BS}) where {T, BS}
    println(io, "MHDProblem{$T, $BS}")
    try
        mp = p.params
        _tree_row(io, "dynamics", "E=$(mp.E), Ra=$(mp.Ra), Pm=$(mp.Pm), Le=$(mp.Le)")
        _tree_row(io, "resolution", "m=$(mp.m), lmax=$(mp.lmax), N=$(mp.N)")
        _tree_row(io, "background field", mp.B0_type; last=true)
    catch
        _tree_row(io, "params type", typeof(p.params); last=true)
    end
end
