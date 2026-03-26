module CrossRecipesBaseExt

using Cross
using RecipesBase

# Eigenvalue spectrum scatter plot
@recipe function f(r::Cross.StabilityResult)
    xlabel --> "Growth rate (σᵣ)"
    ylabel --> "Frequency (σᵢ)"
    seriestype --> :scatter
    markersize --> 6
    markershape --> :circle
    label --> "Eigenvalues ($(length(r.eigenvalues)))"
    real.(r.eigenvalues), imag.(r.eigenvalues)
end

# Growth rate vs parameter sweep
@recipe function f(results::Vector{<:Cross.StabilityResult}; sweep_param=:Ra)
    xlabel --> string(sweep_param)
    ylabel --> "Growth rate"
    seriestype --> :line
    markershape --> :circle
    markersize --> 4
    label --> "Growth rate vs $(sweep_param)"
    xs = [getfield(r.problem.params, sweep_param) for r in results]
    ys = [r.growth_rate for r in results]
    xs, ys
end

end # module
