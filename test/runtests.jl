using Test
using Logging

include("project_deps.jl")
include("legacy_api.jl")

# Existing tests
include("boundary_conditions.jl")
include("mhd_boundary_conditions.jl")
include("mhd_stress_free_bc.jl")
include("meridional_boundary_conditions.jl")
include("chebyshev.jl")
include("sparse_operator.jl")
include("galerkin_radial.jl")
include("mhd_galerkin.jl")
include("onset_api.jl")
include("sh_transform.jl")
include("thermal_wind.jl")
include("triglobal.jl")
include("mean_flow_stability.jl")
include("velocity_reconstruction.jl")

# v2.0 API tests
include("test_types.jl")
include("type_stability.jl")
include("test_validation.jl")
include("test_show.jl")
include("test_problems.jl")
include("test_solve.jl")
include("slepc_backend.jl")
include("dof_ownership.jl")
include("distributed_assembly.jl")
