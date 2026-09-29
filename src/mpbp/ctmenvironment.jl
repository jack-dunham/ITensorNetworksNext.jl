using Dictionaries: Dictionary
using Graphs: edges, neighbors, vertices
using ITensorBase: Index
using NamedGraphs: NamedEdge

struct CTMEnvironment{V, B, E <: MessageCache, C <: MessageCache}
    embedding::PlanarEmbedding{V}
    edgetensors::E
    corners::C
    bonds::Dictionary{NamedEdge{V}, B}
    # Previous subspace basis of each face, read by `face_update!(...; align = true)`.
    gauges::Dict{Int, Any}
end

edgetensor(env::CTMEnvironment, d) = env.edgetensors[NamedEdge(d)]
corner(env::CTMEnvironment, d) = env.corners[NamedEdge(d)]
bond(env::CTMEnvironment, d) = env.bonds[NamedEdge(d)]

function Base.copy(env::CTMEnvironment)
    return CTMEnvironment(
        env.embedding, map(identity, env.edgetensors), map(identity, env.corners),
        copy(env.bonds), copy(env.gauges)
    )
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
    return CTMEnvironment(emb, edgetensors, corners, bonds, Dict{Int, Any}())
end

function environment_tensors(env::CTMEnvironment, v)
    ds = [NamedEdge(w => v) for w in neighbors(env.embedding.graph, v)]
    return [[edgetensor(env, d) for d in ds]; [corner(env, d) for d in ds]]
end

function environment_tensors(env::CTMEnvironment, e::Union{AbstractEdge, Pair})
    emb = env.embedding
    d = NamedEdge(e)
    r = reverse(d)
    return [
        edgetensor(env, d), edgetensor(env, r), corner(env, d),
        corner(env, prev_dart(emb, d)), corner(env, r), corner(env, prev_dart(emb, r)),
    ]
end

# A face is given as its cycle of darts.
function environment_tensors(env::CTMEnvironment, face::AbstractVector{<:AbstractEdge})
    return [corner(env, d) for d in face]
end

function kikuchi_terms(tn, env::CTMEnvironment)
    g = env.embedding.graph
    numerator = (
        vertex_scalars(tn, env, collect(vertices(g))),
        [face_scalar(tn, env, face) for face in env.embedding.faces],
    )
    return numerator, edge_scalars(tn, env, collect(edges(g)))
end
