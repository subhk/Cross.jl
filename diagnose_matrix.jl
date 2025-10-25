#!/usr/bin/env julia
# Diagnose B matrix to find where zeros are

push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using LinearAlgebra
using SparseArrays
using Printf

include("src/SparseOperator.jl")
using .SparseOperator

println("Diagnosing B matrix structure")
println("="^80)

# Small test case
E = 1e-3
χ = 0.35
Pr = 1.0
m = 3
N = 16
lmax = 10
Ra = 1e5

params = SparseOnsetParams(
    E = E,
    Pr = Pr,
    Ra = Ra,
    ricb = χ,
    m = m,
    lmax = lmax,
    symm = 1,
    N = N
)

println("Building operator...")
op = SparseStabilityOperator(params)

println("Assembling matrices...")
A_full, B_full, interior_dofs, info = assemble_sparse_matrices(op)

println()
println("Matrix information:")
println("  Size: $(size(B_full))")
println("  Nonzeros: $(nnz(B_full))")
println("  Interior DOFs returned: $(length(interior_dofs))")
println()

# Extract diagonal of B
B_diag = diag(B_full)

# Find zero and non-zero entries
zero_indices = findall(x -> abs(x) < 1e-15, B_diag)
nonzero_indices = findall(x -> abs(x) >= 1e-15, B_diag)

println("Diagonal of B:")
println("  Total DOFs: $(length(B_diag))")
println("  Nonzero diag entries: $(length(nonzero_indices))")
println("  Zero diag entries: $(length(zero_indices))")
println()

if length(zero_indices) > 0
    println("Zero diagonal entries at indices:")
    for i in zero_indices[1:min(20, end)]
        println("  Index $i: B[$i,$i] = $(B_diag[i])")
    end
    if length(zero_indices) > 20
        println("  ... and $(length(zero_indices) - 20) more")
    end
    println()
end

# Check if interior_dofs actually excludes zeros
println("Checking interior_dofs list:")
zeros_in_interior = filter(i -> abs(B_diag[i]) < 1e-15, interior_dofs)
if length(zeros_in_interior) > 0
    println("  ⚠ WARNING: interior_dofs contains $(length(zeros_in_interior)) indices with zero B diagonal!")
    println("  These should be boundary DOFs, not interior DOFs")
else
    println("  ✓ All interior_dofs have nonzero B diagonal")
end
println()

# Try to manually identify true interior DOFs
true_interior = nonzero_indices
println("True interior DOFs (nonzero B diagonal): $(length(true_interior))")
println()

# Test if we can extract and invert B for true interior
if length(true_interior) > 0
    B_interior = B_full[true_interior, true_interior]
    B_int_diag = diag(B_interior)
    n_zeros = count(x -> abs(x) < 1e-15, B_int_diag)

    println("Extracted interior B matrix:")
    println("  Size: $(size(B_interior))")
    println("  Nonzeros: $(nnz(B_interior))")
    println("  Zero diagonal entries: $n_zeros")

    if n_zeros == 0
        println("  ✓ Interior B matrix is non-singular!")
    else
        println("  ✗ Interior B matrix still has zeros")
    end
end
