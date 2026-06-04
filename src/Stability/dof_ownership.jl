# =============================================================================
#  DOF <-> global-row mapping and PETSc row-ownership queries.
#
#  Pure integer bookkeeping over an `index_map` (1-based Julia row ranges keyed by
#  (ℓ, field)). No PETSc/MPI — the foundation Phase 2+ distributed assembly uses to
#  insert only the rows a rank owns. Conventions: `index_map` ranges are 1-based
#  inclusive `a:b`; PETSc ownership ranges `[rstart, rend)` are 0-based half-open.
# =============================================================================

"""
    row_to_dof(index_map, grow) -> (key, local)

Map a 1-based global row `grow` to its block `key` and 1-based local radial index.
"""
function row_to_dof(index_map::AbstractDict{K,UnitRange{Int}}, grow::Int) where {K}
    for (key, rng) in index_map
        if grow in rng
            return key, grow - first(rng) + 1
        end
    end
    maxrow = isempty(index_map) ? 0 : maximum(last, values(index_map))
    error("row $grow is outside the DOF layout (valid rows 1:$maxrow)")
end

"""
    dof_to_row(index_map, key, local) -> Int

Inverse of `row_to_dof`: 1-based global row for block `key`, 1-based local index.
"""
function dof_to_row(index_map::AbstractDict{K,UnitRange{Int}}, key::K, loc::Int) where {K}
    haskey(index_map, key) || error("unknown DOF block key $key")
    rng = index_map[key]
    (1 <= loc <= length(rng)) ||
        error("local index $loc outside 1:$(length(rng)) for block $key")
    return first(rng) + loc - 1
end

"""
    owned_block_ranges(index_map, rstart, rend) -> Vector{Tuple{K, UnitRange{Int}}}

Blocks whose rows intersect the PETSc ownership range `[rstart, rend)` (0-based,
half-open). For each, returns `(key, owned_local_range)` — the 1-based local radial
indices this rank owns (partial blocks return only the owned slice). Sorted by block
start row for determinism (`index_map` is an unordered Dict).
"""
function owned_block_ranges(index_map::AbstractDict{K,UnitRange{Int}},
                            rstart::Int, rend::Int) where {K}
    out = Tuple{K,UnitRange{Int}}[]
    for (key, rng) in index_map
        a = first(rng)
        b = last(rng)
        lo = max(a - 1, rstart)
        hi = min(b, rend)
        if lo < hi
            push!(out, (key, (lo - a + 2):(hi - a + 1)))
        end
    end
    sort!(out; by = t -> first(index_map[t[1]]))
    return out
end
