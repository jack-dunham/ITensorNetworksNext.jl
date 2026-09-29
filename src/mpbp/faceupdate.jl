using Dictionaries: set!
using Graphs: dst, neighbors, src
using ITensorBase: Index, name, names
using LinearAlgebra: Diagonal, I, eigen, inv, norm
using NamedGraphs: NamedEdge
using TensorAlgebra: matricize

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

cut_names(tn, env::CTMEnvironment, d) = (linknames(tn, d)..., name(bond(env, reverse(d))))

# Contracts the tensors at `v = src(ds[i])` outside face `f`: the edge tensors not on the face and
# the corners whose wedge touches no face edge.
function corner_transfer_matrix(tn, env::CTMEnvironment, f::Int, i::Int)
    emb = env.embedding
    ds = emb.faces[f]
    dprev, dnext = ds[mod1(i - 1, length(ds))], ds[i]
    v = src(dnext)
    onface(w) = w ∈ (src(dprev), dst(dnext))
    ws = neighbors(emb.graph, v)
    ts = [
        [tn[v]];
        [edgetensor(env, w => v) for w in ws if !onface(w)];
        [
            corner(env, w => v) for w in ws
                if !onface(w) && !onface(dst(next_dart(emb, w => v)))
        ]
    ]
    C = contract_network(ts)
    # A cut bond that no included tensor carries has dimension 1 and is attached as a unit leg.
    for r in (reverse(dprev), reverse(dnext))
        β = bond(env, r)
        if name(β) ∉ names(C)
            @assert length(β) == 1
            C = C * ones(eltype(C), (β,))
        end
    end
    @assert issetequal(
        names(C),
        (cut_names(tn, env, dprev)..., cut_names(tn, env, dnext)...)
    )
    return C
end

# `X` equals `A` contracted with corner `c` over the one leg they share; returns `A`.
function peel(X, c)
    s = only(intersect(names(X), names(c)))
    k = only(setdiff(names(c), (s,)))
    M = matricize(c, (k,), (s,))
    return X * fromarray(inv(M), (s, k), namedsize(c, (s, k)))
end

"""
    face_update!(env, tn, f; maxdim, alg, frozen = false, align = false) -> env

Replace the corners, bonds and edge tensors of face `f` with the MP-BP solution of the face
given the rest of `env`. Every corner of `f` is set to the identity except the one on its
second-to-last dart, which holds the eigenvalues of the face's corner transfer matrix product.

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
    ds = env.embedding.faces[f]
    m = length(ds)
    m ≥ 3 || throw(ArgumentError("Face $f has $m darts; `face_update!` needs at least 3."))
    cuts = [cut_names(tn, env, d) for d in ds]
    Cs = [corner_transfer_matrix(tn, env, f, i) for i in 1:m]
    Cm = [
        matricize(C / norm(C), cutprev, cut)
            for (C, cutprev, cut) in zip(Cs, circshift(cuts, 1), cuts)
    ]
    VR, VL, Λ = face_subspace(env, f, Cm; maxdim, alg, frozen, align)
    VRs, VLs = propagate_bases(Cm, VR, VL, Λ)
    β = frozen ? [bond(env, d) for d in ds] : [Index(size(Λ, 1)) for _ in ds]
    set_edgetensors!(
        env,
        ds,
        [namedsize(C, cut) for (C, cut) in zip(Cs, cuts)],
        cuts,
        VRs,
        VLs,
        β
    )
    set_corners!(env, ds, Λ, β; frozen)
    return env
end

# Dominant subspace of the product of `Cm`, rotated onto the previous basis of `f` if `align`.
function face_subspace(env::CTMEnvironment, f::Int, Cm; maxdim, alg, frozen, align)
    χfrozen = length(bond(env, first(env.embedding.faces[f])))
    VR, VL, λ = invariant_subspace(alg, foldl(*, Cm), frozen ? χfrozen : maxdim)
    if frozen && length(λ) != χfrozen
        throw(
            ArgumentError(
                "Face $f keeps $(length(λ)) eigenvalues but its frozen bonds have dimension $χfrozen."
            )
        )
    end
    Λ = Matrix(Diagonal(λ))
    align || return VR, VL, Λ
    reference = get(env.gauges, f, nothing)
    if !isnothing(reference) && size(reference) == size(VR)
        g = VL * reference
        VR, VL, Λ = VR * g, g \ VL, g \ (Λ * g)
    end
    env.gauges[f] = VR
    return VR, VL, Λ
end

# Carries the bases at dart `m` around the face; `Λ` is divided out at dart `m - 1`, whose
# corner holds it.
function propagate_bases(Cm, VR, VL, Λ)
    m = length(Cm)
    VRs = Vector{Matrix{eltype(VR)}}(undef, m)
    VLs = Vector{Matrix{eltype(VL)}}(undef, m)
    VRs[m], VLs[m] = VR, VL
    for j in m:-1:2
        VRs[j - 1] = Cm[j] * VRs[j]
    end
    for j in 1:(m - 1)
        prev = VLs[mod1(j - 1, m)]
        VLs[j] = (j == m - 1 ? Λ \ prev : prev) * Cm[j]
    end
    return VRs, VLs
end

# Writes the edge tensors on both sides of each dart of the face.
function set_edgetensors!(env::CTMEnvironment, ds, cutdims, cuts, VRs, VLs, β)
    emb = env.embedding
    χ = length(first(β))
    for (j, (d, βprev, βnext)) in enumerate(zip(ds, circshift(β, 1), circshift(β, -1)))
        r = reverse(d)
        slice(M, b) = fromarray(M, (cuts[j]..., name(b)), (cutdims[j]..., χ))
        env.edgetensors[r] = peel(slice(VRs[j], βprev), corner(env, r))
        env.edgetensors[d] =
            peel(slice(transpose(VLs[j]), βnext), corner(env, prev_dart(emb, r)))
    end
    return env
end

function set_corners!(env::CTMEnvironment, ds, Λ, β; frozen)
    χ = size(Λ, 1)
    for (j, (d, b, bnext)) in enumerate(zip(ds, β, circshift(β, -1)))
        cj = j == length(ds) - 1 ? Λ : Matrix{eltype(Λ)}(I, χ, χ)
        env.corners[d] = fromarray(cj, (name(b), name(bnext)), (χ, χ))
        frozen || set!(env.bonds, d, b)
    end
    return env
end
