using Dictionaries: set!
using Graphs: dst, src
using ITensorBase: Index, name, names
using LinearAlgebra: Diagonal, I, eigen, inv, norm
using TensorAlgebra: matricize, unmatricize

@kwdef struct DenseEig <: AbstractAlgorithm
    rtol::Float64 = 1.0e-12
    degeneracy_rtol::Float64 = 1.0e-10
end

function invariant_subspace end

function default_algorithm(::typeof(invariant_subspace), ::Type{<:Tuple}; kwargs...)
    return DenseEig(; kwargs...)
end

"""
    invariant_subspace(alg::DenseEig, Λ, maxdim) -> (VR, VL, λ)

Dominant invariant subspace of `Λ`: `Λ * VR ≈ VR * Diagonal(λ)`, `VL * Λ ≈ Diagonal(λ) * VL`,
`VL * VR ≈ I`. Eigenvalues of equal modulus are kept or dropped together.
"""
function invariant_subspace(alg::DenseEig, Λ::AbstractMatrix, maxdim::Integer)
    F = eigen(Λ)
    p = sortperm(abs.(F.values); rev = true)
    vals, vecs = F.values[p], F.vectors[:, p]
    scale = abs(first(vals))
    χ = min(maxdim, count(λ -> abs(λ) > alg.rtol * scale, vals))
    while 0 < χ < length(vals) &&
            abs(abs(vals[χ]) - abs(vals[χ + 1])) ≤ alg.degeneracy_rtol * abs(vals[χ])
        χ -= 1
    end
    χ > 0 || throw(
        ArgumentError(
            "`maxdim = $maxdim` splits the dominant eigenvalue multiplet; raise `maxdim`."
        )
    )
    VL = transpose(transpose(vecs) \ Matrix{eltype(vecs)}(I, size(vecs, 1), χ))
    return vecs[:, 1:χ], VL, vals[1:χ]
end

cut_inds(tn, env::CTMEnvironment, d) = (linkinds(tn, d)..., bond(env, reverse(d)))

function corner_transfer_matrix(tn, env::CTMEnvironment, f::Int, i::Int)
    emb = env.embedding
    ds = emb.faces[f]
    m = length(ds)
    dprev, dnext = ds[mod1(i - 1, m)], ds[i]
    v = src(dnext)
    a, b = src(dprev), dst(dnext)
    C = contract_network([[tn[v]]; environment_tensors(env, v; exclude = (a, b))])
    elt = eltype(C)
    for r in (reverse(dprev), reverse(dnext))
        β = bond(env, r)
        if name(β) ∉ names(C)
            @assert length(β) == 1
            C = C * ones(elt, (β,))
        end
    end
    @assert issetequal(
        names(C),
        name.((cut_inds(tn, env, dprev)..., cut_inds(tn, env, dnext)...))
    )
    return C
end

# `X` equals `A` contracted with corner `c` over `k`; returns `A`, whose leg `s` becomes `k`.
function peel(X, c, s, k)
    return X * unmatricize(inv(matricize(c, (k,), (s,))), (s,), (k,))
end

"""
    face_update!(env, tn, f; maxdim, alg, frozen = false, align = false) -> env

Replace the corners, bonds and edge tensors of face `f` with the MP-BP solution of the face
given the rest of `env`.

With `frozen = true` the face keeps its bond indices, so the kept subspace must have their
dimension. With `align = true` the new subspace basis is rotated onto the one the previous
update of `f` stored, which keeps the tensors continuous between updates when eigenvalues
share a modulus; the eigenvalue corner is then a full matrix.
"""
function face_update!(
        env::CTMEnvironment,
        tn,
        f::Int;
        maxdim::Integer,
        alg::AbstractAlgorithm,
        frozen::Bool = false,
        align::Bool = false
    )
    emb = env.embedding
    ds = emb.faces[f]
    m = length(ds)
    m ≥ 3 || throw(ArgumentError("Face $f has $m darts; `face_update!` needs at least 3."))
    cuts = [cut_inds(tn, env, d) for d in ds]
    Cs = [corner_transfer_matrix(tn, env, f, i) for i in 1:m]
    Cm = [matricize(Cs[i] / norm(Cs[i]), cuts[mod1(i - 1, m)], cuts[i]) for i in 1:m]
    χfrozen = length(bond(env, first(ds)))
    VR, VL, λ = invariant_subspace(alg, foldl(*, Cm), frozen ? χfrozen : maxdim)
    χ = length(λ)
    frozen && χ != χfrozen &&
        throw(
        ArgumentError(
            "Face $f keeps $χ eigenvalues but its frozen bonds have dimension $χfrozen."
        )
    )
    Λ = Matrix(Diagonal(λ))
    if align
        reference = get(env.gauges, f, nothing)
        if !isnothing(reference) && size(reference) == size(VR)
            g = VL * reference
            VR, VL, Λ = VR * g, g \ VL, g \ (Λ * g)
        end
        env.gauges[f] = VR
    end
    VRs = Vector{Matrix{eltype(VR)}}(undef, m)
    VLs = Vector{Matrix{eltype(VL)}}(undef, m)
    VRs[m], VLs[m] = VR, VL
    for j in m:-1:2
        VRs[j - 1] = Cm[j] * VRs[j]
    end
    VLs[1] = VLs[m] * Cm[1]
    for j in 2:(m - 2)
        VLs[j] = VLs[j - 1] * Cm[j]
    end
    VLs[m - 1] = (Λ \ VLs[m - 2]) * Cm[m - 1]
    β = frozen ? [bond(env, d) for d in ds] : [Index(χ) for _ in 1:m]
    for j in 1:m
        d, r = ds[j], reverse(ds[j])
        P = unmatricize(VRs[j], cuts[j], (β[mod1(j - 1, m)],))
        Q = unmatricize(transpose(VLs[j]), cuts[j], (β[mod1(j + 1, m)],))
        env.edgetensors[r] =
            peel(P, corner(env, r), bond(env, r), bond(env, next_dart(emb, r)))
        pr = prev_dart(emb, r)
        env.edgetensors[d] = peel(Q, corner(env, pr), bond(env, r), bond(env, pr))
    end
    elt = promote_type(eltype(VR), eltype(VL), eltype(Λ))
    for j in 1:m
        cj = j == m - 1 ? Matrix{elt}(Λ) : Matrix{elt}(I, χ, χ)
        env.corners[ds[j]] = unmatricize(cj, (β[j],), (β[mod1(j + 1, m)],))
        frozen || set!(env.bonds, ds[j], β[j])
    end
    return env
end
