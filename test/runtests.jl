using Test
using Logging

include("project_deps.jl")

# Existing tests
include("boundary_conditions.jl")
include("chebyshev.jl")
include("sparse_operator.jl")
include("thermal_wind.jl")
include("triglobal.jl")

# v2.0 API tests
include("test_types.jl")
include("type_stability.jl")
include("test_validation.jl")
include("test_show.jl")
include("test_problems.jl")
include("test_solve.jl")
