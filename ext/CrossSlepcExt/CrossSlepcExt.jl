module CrossSlepcExt

using Cross
using SparseArrays
using PetscWrap
using SlepcWrap

# PetscWrap does `using MPI` internally, so its MPI module binding is reachable as
# PetscWrap.MPI. Aliasing it (rather than adding MPI as a separate weakdep) also
# guarantees we use the exact same MPI.Comm type PetscWrap's ccalls expect.
const MPI = PetscWrap.MPI

include("raw_petsc.jl")

const _INITIALIZED = Ref(false)

function _slepc_init!(opts::AbstractString="")
    if !_INITIALIZED[]
        SlepcInitialize(String(opts))
        _INITIALIZED[] = true
    end
    return nothing
end

function _slepc_finalize!()
    if _INITIALIZED[]
        SlepcFinalize()
        _INITIALIZED[] = false
    end
    return nothing
end

# Build a distributed MPIAIJ PETSc matrix from the full (replicated) Julia CSC,
# inserting only this rank's owned rows.
function _to_petsc_dist(M::SparseMatrixCSC, n::Int)
    mat = MatCreate(MPI.COMM_WORLD)
    MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, n, n)
    MatSetFromOptions(mat)
    rstart, rend = MatGetOwnershipRange(mat)            # 0-based, half-open
    d, o = Cross._petsc_owned_nnz(M, Int(rstart), Int(rend))
    PI = PetscWrap.PetscInt
    MatMPIAIJSetPreallocation(mat, PI(0), PI.(d), PI(0), PI.(o))
    rows = rowvals(M); vals = nonzeros(M)
    @inbounds for col in 1:size(M, 2)
        for k in nzrange(M, col)
            r0 = rows[k] - 1
            if rstart <= r0 < rend
                MatSetValue(mat, r0, col - 1, PetscScalar(vals[k]), INSERT_VALUES)
            end
        end
    end
    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY)
    return mat
end

"""
Distributed SLEPc solve of `A x = σ B x` over `MPI.COMM_WORLD`. Replicated Julia
assembly (each rank holds full `A`/`B`, inserts only owned rows). MUMPS shift-invert
comes from the option string set in `slepc_init!`. Returns the Cross contract
`(eigenvalues, eigenvectors, info)`: eigenvalues on all ranks; eigenvectors full
`n×nev` on rank 0, empty `n×0` on workers. Requires a complex-scalar PETSc build.
"""
function _slepc_solve(A::SparseMatrixCSC, B::SparseMatrixCSC;
                      nev::Int, sigma, which::Symbol, selection::Symbol,
                      tol::Float64, maxiter::Int, verbosity::Int=0)
    _INITIALIZED[] || error("call Cross.slepc_init!() once before a :slepc solve")
    PetscScalar <: Real &&
        error("PETSc/SLEPc must be built with complex scalars (--with-scalar-type=complex)")
    size(A) == size(B) || throw(DimensionMismatch("A and B must match"))
    n = size(A, 1)

    target = sigma === nothing ?
        (which === :LR ? ComplexF64(10, 0) :
         which === :LI ? ComplexF64(0, 10) : ComplexF64(1, 0)) :
        ComplexF64(sigma)

    Amat = _to_petsc_dist(A, n)
    Bmat = _to_petsc_dist(B, n)

    eps = EPSCreate(MPI.COMM_WORLD)
    EPSSetOperators(eps, Amat, Bmat)
    _eps_set_dimensions(eps, nev)
    EPSSetTarget(eps, PetscScalar(target))
    EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE)
    EPSSetFromOptions(eps)        # GNHEP + sinvert + MUMPS come from slepc_init! opts
    EPSSetUp(eps)
    EPSSolve(eps)

    nconv = EPSGetConverged(eps)
    nout = min(nconv, nev)
    nout == 0 && (EPSDestroy(eps); MatDestroy(Amat); MatDestroy(Bmat);
                  error("SLEPc returned no converged eigenpairs"))

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    vals = Vector{ComplexF64}(undef, nout)
    vecs = rank == 0 ? Matrix{ComplexF64}(undef, n, nout) : Matrix{ComplexF64}(undef, n, 0)
    vr, vi = MatCreateVecs(Amat)
    for j in 0:(nout - 1)
        vpr, vpi, vecr, veci = EPSGetEigenpair(eps, j, vr, vi)
        vals[j + 1] = ComplexF64(vpr, vpi)            # collective: identical all ranks
        full = _vec_scatter_to_zero(vecr)             # length n on rank 0, else 0
        rank == 0 && (vecs[:, j + 1] .= full)
    end

    info = Dict{String,Any}("solver" => :slepc, "strategy" => :shift_invert,
        "target" => target, "nconv" => nconv, "selection" => selection,
        "ranks" => MPI.Comm_size(MPI.COMM_WORLD))

    EPSDestroy(eps); MatDestroy(Amat); MatDestroy(Bmat)   # NOT SlepcFinalize (explicit lifecycle)

    perm = _sort_indices_local(vals, selection)
    return vals[perm], (size(vecs, 2) == 0 ? vecs : vecs[:, perm]), info
end

function _sort_indices_local(ev::AbstractVector{<:Complex}, selection::Symbol)
    selection === :maxreal      ? sortperm(real.(ev); rev=true) :
    selection === :minabs       ? sortperm(abs.(ev)) :
    selection === :closest_real ? sortperm(abs.(real.(ev))) :
    error("Unknown selection strategy $(selection)")
end

function __init__()
    Cross._SLEPC_SOLVER[]   = _slepc_solve
    Cross._SLEPC_INIT[]     = _slepc_init!
    Cross._SLEPC_FINALIZE[] = _slepc_finalize!
    return nothing
end

end # module
