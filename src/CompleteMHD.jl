# =============================================================================
#  Complete MHD Module for Cross.jl
#
#  This is the main MHD module that includes all sub-components.
#  Use this file to load the full MHD functionality.
#
#  Usage:
#    include("src/CompleteMHD.jl")
#    using .CompleteMHD
# =============================================================================

module CompleteMHD

using LinearAlgebra
using SparseArrays
using Printf
using SpecialFunctions

# Load ultraspherical spectral methods
push!(LOAD_PATH, @__DIR__)
include("UltrasphericalSpectral.jl")
using .UltrasphericalSpectral

# Include MHD operator definitions
include("MHDOperator.jl")
using .MHDOperator
using .MHDOperator: background_operator

# Include hydrodynamic operators from SparseOperator (needed for velocity and temperature)
include("SparseOperator.jl")

# Include MHD operator functions
include("MHDOperatorFunctions.jl")

# Include MHD assembly
include("MHDAssembly.jl")

# Export main types and functions
export MHDParams,
       MHDStabilityOperator,
       BackgroundField,
       no_field, axial, dipole,
       assemble_mhd_matrices,
       operator_lorentz_poloidal_diagonal,
       operator_lorentz_poloidal_offdiag,
       operator_lorentz_toroidal,
       operator_lorentz_toroidal_from_bpol,
       operator_lorentz_toroidal_from_btor,
       operator_induction_poloidal_from_u,
       operator_induction_poloidal_from_v,
       operator_induction_toroidal_from_u,
       operator_induction_toroidal_from_v,
       operator_magnetic_diffusion_poloidal,
       operator_magnetic_diffusion_toroidal,
       spherical_bessel_j_logderiv

println("CompleteMHD module loaded successfully")
println("  - MHD parameters and operators")
println("  - Lorentz force coupling")
println("  - Induction equation")
println("  - Magnetic diffusion")
println("  - Background fields: axial, dipole")

end  # module CompleteMHD
