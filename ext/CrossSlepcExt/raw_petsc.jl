# Raw ccall bindings for primitives SlepcWrap 0.1.3 / PetscWrap 0.1.5 do not wrap.
# Signatures follow PETSc/SLEPc C and PetscWrap's own ccall convention
# (CVec/CMat == Ptr{Cvoid}; wrappers cconvert to their handle). UNTESTED here (no
# PETSc) — confirm against the installed PETSc/SLEPc headers on the cluster.

const CVecScatter = Ptr{Cvoid}
const SCATTER_FORWARD = Cint(0)
const _INSERT_VALUES_C = Cint(1)   # PETSc InsertMode INSERT_VALUES

"""Set the requested eigenpair count on an EPS (SlepcWrap 0.1.3 has no wrapper).
`EPSSetDimensions(eps, nev, ncv=PETSC_DECIDE, mpd=PETSC_DECIDE)`."""
function _eps_set_dimensions(eps, nev::Integer)
    PD = PetscWrap.PETSC_DECIDE
    err = ccall((:EPSSetDimensions, SlepcWrap.libslepc), PetscWrap.PetscErrorCode,
                (Ptr{Cvoid}, PetscWrap.PetscInt, PetscWrap.PetscInt, PetscWrap.PetscInt),
                eps.ptr[], PetscWrap.PetscInt(nev), PD, PD)
    @assert iszero(err)
    return nothing
end

"""Gather a distributed PETSc vector to rank 0 as a `Vector{ComplexF64}`: full
length-`n` on rank 0, empty elsewhere. Wraps VecScatterCreateToZero / Begin / End /
VecGetArray / VecScatterDestroy / VecDestroy."""
function _vec_scatter_to_zero(v::PetscWrap.PetscVec)
    ctx = Ref{CVecScatter}()
    seq = Ref{PetscWrap.CVec}()
    @assert iszero(ccall((:VecScatterCreateToZero, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
        (PetscWrap.CVec, Ptr{CVecScatter}, Ptr{PetscWrap.CVec}), v, ctx, seq))
    @assert iszero(ccall((:VecScatterBegin, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
        (CVecScatter, PetscWrap.CVec, PetscWrap.CVec, Cint, Cint),
        ctx[], v, seq[], _INSERT_VALUES_C, SCATTER_FORWARD))
    @assert iszero(ccall((:VecScatterEnd, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
        (CVecScatter, PetscWrap.CVec, PetscWrap.CVec, Cint, Cint),
        ctx[], v, seq[], _INSERT_VALUES_C, SCATTER_FORWARD))

    nref = Ref{PetscWrap.PetscInt}()
    ccall((:VecGetSize, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
          (PetscWrap.CVec, Ref{PetscWrap.PetscInt}), seq[], nref)
    n = Int(nref[])
    out = Vector{ComplexF64}(undef, n)
    if n > 0
        aref = Ref{Ptr{PetscWrap.PetscScalar}}()
        ccall((:VecGetArray, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
              (PetscWrap.CVec, Ref{Ptr{PetscWrap.PetscScalar}}), seq[], aref)
        arr = unsafe_wrap(Array, aref[], n; own=false)
        out .= ComplexF64.(arr)
        ccall((:VecRestoreArray, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
              (PetscWrap.CVec, Ref{Ptr{PetscWrap.PetscScalar}}), seq[], aref)
    end
    ccall((:VecScatterDestroy, PetscWrap.libpetsc), PetscWrap.PetscErrorCode, (Ptr{CVecScatter},), ctx)
    ccall((:VecDestroy, PetscWrap.libpetsc), PetscWrap.PetscErrorCode, (Ptr{PetscWrap.CVec},), seq)
    return out
end
