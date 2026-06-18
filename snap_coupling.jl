# Bit-identical snapshot harness for the triglobal mode-coupling assembly.
# Usage:  julia --project=. snap_coupling.jl save    # capture golden coupling_ops
#         julia --project=. snap_coupling.jl check   # compare current vs golden
#
# Populates a BasicState3D with deterministic non-zero coeffs so every
# add_*_coupling! term fires, then builds coupling_ops and (de)serializes it.
using Cross
using Serialization
using Random

const SNAP = "/tmp/triglobal_coupling_snap.jls"

function build_coupling_ops()
    T = Float64
    Nr = 12; χ = T(0.35)
    lmax_bs = 2; mmax_bs = 1
    cd = ChebyshevDiffn(Nr, T[χ, one(T)], 1)
    r = cd.x
    Random.seed!(20260618)
    # Deterministic non-zero coeffs for every (ℓ,m), 0≤m≤mmax_bs, m≤ℓ≤lmax_bs.
    mk() = begin
        d = Dict{Tuple{Int,Int}, Vector{T}}()
        for m in 0:mmax_bs, ℓ in m:lmax_bs
            d[(ℓ, m)] = T(0.01) .* (1 .+ sin.(T(ℓ + 1) .* r .+ T(m)))
        end
        d
    end
    θ = mk(); dθ = mk(); ur = mk(); uθ = mk(); uφ = mk(); dur = mk(); duθ = mk(); duφ = mk()
    bs3d = BasicState3D{T}(
        lmax_bs = lmax_bs, mmax_bs = mmax_bs, Nr = Nr, r = r,
        theta_coeffs = θ, dtheta_dr_coeffs = dθ,
        ur_coeffs = ur, utheta_coeffs = uθ, uphi_coeffs = uφ,
        dur_dr_coeffs = dur, dutheta_dr_coeffs = duθ, duphi_dr_coeffs = duφ)
    params = Cross.TriglobalParams(E=T(1e-3), Pr=one(T), Ra=T(100.0), χ=χ,
        m_range = -1:1, lmax = 4, Nr = Nr, basic_state_3d = bs3d)
    problem = Cross.setup_coupled_mode_problem(params)
    smo = Cross.build_single_mode_operators(problem, false)
    cops = Cross.build_mode_coupling_operators(problem, smo, false)
    return cops
end

mode = length(ARGS) >= 1 ? ARGS[1] : "check"
cops = build_coupling_ops()
npairs = length(cops)
nnz_tot = sum(count(!iszero, C) for C in values(cops); init=0)
println("coupling_ops: $npairs pairs, $nnz_tot nonzeros total")

if mode == "save"
    serialize(SNAP, cops)
    println("SAVED golden snapshot → $SNAP")
else
    golden = deserialize(SNAP)
    @assert Set(keys(golden)) == Set(keys(cops)) "key mismatch: golden=$(sort(collect(keys(golden)))) cur=$(sort(collect(keys(cops))))"
    for k in keys(golden)
        @assert size(golden[k]) == size(cops[k]) "size mismatch at $k: $(size(golden[k])) vs $(size(cops[k]))"
    end
    diffs = [isempty(golden[k]) ? 0.0 : maximum(abs.(golden[k] .- cops[k])) for k in keys(golden)]
    maxdiff = isempty(diffs) ? 0.0 : maximum(diffs)
    println("maxdiff vs golden = ", maxdiff)
    println(maxdiff == 0.0 ? "BIT-IDENTICAL ✓" : (maxdiff < 1e-12 ? "within tol (reorder) ✓" : "DIVERGED ✗"))
end
