using Dictionaries: Dictionary
using Graphs: dst, edges, neighbors, vertices
using ITensorBase: Index
using NamedGraphs: NamedEdge, all_edges

struct CTMEnvironment{V, B, E <: MessageCache, C <: MessageCache}
    embedding::PlanarEmbedding{V}
    edgetensors::E
    corners::C
    bonds::Dictionary{NamedEdge{V}, B}
end

edgetensor(env::CTMEnvironment, edge) = env.edgetensors[NamedEdge(edge)]
corner(env::CTMEnvironment, edge) = env.corners[NamedEdge(edge)]
bond(env::CTMEnvironment, edge) = env.bonds[NamedEdge(edge)]
# The two bonds of `c[edge]`, as the (rows, columns) of the corner seen as a matrix.
function corner_bonds(env::CTMEnvironment, edge)
    return (bond(env, edge),), (bond(env, next_edge(env.embedding, edge)),)
end

function Base.copy(env::CTMEnvironment)
    return CTMEnvironment(
        env.embedding, map(identity, env.edgetensors), map(identity, env.corners),
        copy(env.bonds)
    )
end

"""
    ctm_environment(tn, embedding, messages) -> CTMEnvironment

The χ = 1 environment whose edge tensors are the BP `messages` and whose corners are all 1.
"""
function ctm_environment(tn, embedding::PlanarEmbedding, messages)
    embedding_edges = collect(all_edges(embedding.graph))
    bonds = Dictionary(embedding_edges, [Index(1) for _ in embedding_edges])
    elt = eltype(messages[first(embedding_edges)])

    edgetensors = messagecache(embedding_edges) do edge
        return messages[edge] * ones(elt, (bonds[next_edge(embedding, edge)],)) *
            ones(elt, (bonds[prev_edge(embedding, reverse(edge))],))
    end
    corners = messagecache(embedding_edges) do edge
        return ones(elt, (bonds[edge], bonds[next_edge(embedding, edge)]))
    end

    return CTMEnvironment(embedding, edgetensors, corners, bonds)
end

# A corner `c[w => v]` touches `w` and the far end of `next_edge(w => v)`.
function environment_tensors(env::CTMEnvironment, vertex; exclude = ())
    embedding = env.embedding
    incoming = [
        NamedEdge(neighbor => vertex)
            for neighbor in neighbors(embedding.graph, vertex) if neighbor ∉ exclude
    ]

    return [
        [edgetensor(env, edge) for edge in incoming];
        [
            corner(env, edge) for
                edge in incoming if dst(next_edge(embedding, edge)) ∉ exclude
        ]
    ]
end

function environment_tensors(env::CTMEnvironment, edge::Union{AbstractEdge, Pair})
    embedding = env.embedding
    forward = NamedEdge(edge)
    backward = reverse(forward)
    return [
        edgetensor(env, forward), edgetensor(env, backward),
        corner(env, forward), corner(env, prev_edge(embedding, forward)),
        corner(env, backward), corner(env, prev_edge(embedding, backward)),
    ]
end

# A face is given as its cycle of directed edges.
function environment_tensors(env::CTMEnvironment, face::AbstractVector{<:AbstractEdge})
    return view(env.corners, face)
end

function kikuchi_terms(tn, env::CTMEnvironment)
    graph = env.embedding.graph
    numerator = (
        vertex_scalars(tn, env, collect(vertices(graph))),
        [face_scalar(tn, env, face) for face in env.embedding.faces],
    )
    return numerator, edge_scalars(tn, env, collect(edges(graph)))
end
