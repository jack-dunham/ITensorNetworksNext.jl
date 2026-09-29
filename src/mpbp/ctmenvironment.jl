using Dictionaries: Dictionary
using Graphs: edges, neighbors, vertices
using ITensorBase: Index, NamedTensor, align, names, unnamed
using NamedGraphs: NamedEdge

struct CTMEnvironment{V, B, E <: MessageCache, C <: MessageCache}
    embedding::PlanarEmbedding{V}
    edgetensors::E
    corners::C
    bonds::Dictionary{NamedEdge{V}, B}
end

edgetensor(env::CTMEnvironment, d) = env.edgetensors[NamedEdge(d)]
corner(env::CTMEnvironment, d) = env.corners[NamedEdge(d)]
bond(env::CTMEnvironment, d) = env.bonds[NamedEdge(d)]

function Base.copy(env::CTMEnvironment)
    return CTMEnvironment(
        env.embedding, map(identity, env.edgetensors), map(identity, env.corners),
        copy(env.bonds)
    )
end

fromarray(M, names, dims) = NamedTensor(reshape(M, dims...), names)
function namedsize(t, ns)
    rest = setdiff(names(t), ns)
    return size(unnamed(align(t, (ns..., rest...))))[1:length(ns)]
end

"""
    ctm_environment(tn, emb, messages) -> CTMEnvironment

The χ = 1 environment whose edge tensors are the BP `messages` and whose corners are all 1.
"""
function ctm_environment(tn, emb::PlanarEmbedding, messages)
    ds = darts(emb)
    bonds = Dictionary(ds, [Index(1) for _ in ds])
    elt = eltype(messages[first(ds)])
    edgetensors = messagecache(ds) do d
        m = messages[d]
        return m * ones(elt, (bonds[next_dart(emb, d)],)) *
            ones(elt, (bonds[prev_dart(emb, reverse(d))],))
    end
    corners = messagecache(d -> ones(elt, (bonds[d], bonds[next_dart(emb, d)])), ds)
    return CTMEnvironment(emb, edgetensors, corners, bonds)
end

function vertex_term(tn, env::CTMEnvironment, v; tensor = tn[v])
    ds = [NamedEdge(w => v) for w in neighbors(env.embedding.graph, v)]
    return contract_network(
        [[tensor]; [edgetensor(env, d) for d in ds]; [corner(env, d) for d in ds]]
    )[]
end

function edge_term(env::CTMEnvironment, e)
    emb = env.embedding
    d = NamedEdge(e)
    r = reverse(d)
    ts = [
        edgetensor(env, d), edgetensor(env, r), corner(env, d),
        corner(env, prev_dart(emb, d)), corner(env, r), corner(env, prev_dart(emb, r)),
    ]
    return contract_network(ts)[]
end

function face_term(env::CTMEnvironment, f::Int)
    return contract_network(corner.(Ref(env), env.embedding.faces[f]))[]
end

function bethe_free_entropy(tn, env::CTMEnvironment)
    g = env.embedding.graph
    vs = [vertex_term(tn, env, v) for v in vertices(g)]
    es = [edge_term(env, e) for e in edges(g)]
    fs = [face_term(env, f) for f in eachindex(env.embedding.faces)]
    any(iszero, es) && return -Inf
    return sumlog(vs) + sumlog(fs) - sumlog(es)
end

function expect(tn, env::CTMEnvironment, v, tensor)
    return vertex_term(tn, env, v; tensor) / vertex_term(tn, env, v)
end
