module CrossMakieExt

using Cross
using Makie

# Interactive eigenvalue spectrum
function Cross.eigenspectrum(r::Cross.StabilityResult; figure_kwargs...)
    fig = Figure(; figure_kwargs...)
    ax = Axis(fig[1,1],
        xlabel="σᵣ (growth rate)",
        ylabel="σᵢ (frequency)",
        title="Eigenvalue Spectrum")
    scatter!(ax, real.(r.eigenvalues), imag.(r.eigenvalues),
        markersize=12, color=:blue)
    DataInspector(fig)
    fig
end

# Meridional slice contour
function Cross.plot_meridional(r::Cross.StabilityResult, mode_index::Int;
                                field::Symbol=:temperature, npoints::Int=100)
    fig = Figure()
    ax = Axis(fig[1,1],
        xlabel="r",
        ylabel="θ",
        title="Meridional slice: $(field), mode $mode_index")
    # Field reconstruction requires eigenvector reshaping based on problem structure
    # This will be fully implemented when the eigenvector layout is stabilized
    @warn "plot_meridional: field reconstruction pending full eigenvector layout specification"
    fig
end

# Radial profile per harmonic degree
function Cross.plot_radial(r::Cross.StabilityResult, mode_index::Int;
                            field::Symbol=:poloidal)
    fig = Figure()
    ax = Axis(fig[1,1],
        xlabel="r",
        ylabel="|amplitude|",
        title="Radial profile: $(field), mode $mode_index")
    # Per-l decomposition requires knowledge of block structure
    @warn "plot_radial: per-l decomposition pending full eigenvector layout specification"
    fig
end

end # module
