using Dictionaries: Dictionary
using Graphs: dst, edges, neighbors, vertices
using ITensorBase: Index, commoninds
using NamedGraphs: NamedEdge, all_edges

struct CTMEnvironment{V, E <: MessageCache, C <: MessageCache}
    embedding::PlanarEmbedding{V}
    edgetensors::E
    corners::C
    # Each face's `(right_basis, left_basis)` from its last update, which the next one aligns
    # onto and `SubspaceIteration` starts from.
    bases::Dict{NamedFace{V}, Tuple{AbstractMatrix, AbstractMatrix}}
end

edgetensor(env::CTMEnvironment, edge) = env.edgetensors[NamedEdge(edge)]
corner(env::CTMEnvironment, edge) = env.corners[NamedEdge(edge)]
# `c[edge]` and the corner before it in the same face share exactly the bond of `edge`.
function bond(env::CTMEnvironment, edge)
    previous = prevedge(env.embedding, edge)
    return only(commoninds(corner(env, edge), corner(env, previous)))
end

# Two-index tensor over `row` and `column` with `diagonal` on its diagonal; every corner is one.
function diagonal_tensor(diagonal, row, column)
    tensor = zeros(eltype(diagonal), (row, column))
    for (k, value) in enumerate(diagonal)
        tensor[row => k, column => k] = value
    end
    return tensor
end

function corner_diagonal(env::CTMEnvironment, edge)
    row, column = bond(env, edge), bond(env, nextedge(env.embedding, edge))
    tensor = corner(env, edge)
    return [tensor[row => k, column => k] for k in 1:length(row)]
end

function Base.copy(env::CTMEnvironment)
    return CTMEnvironment(
        env.embedding, map(identity, env.edgetensors), map(identity, env.corners),
        copy(env.bases)
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
        return messages[edge] * ones(elt, (bonds[nextedge(embedding, edge)],)) *
            ones(elt, (bonds[prevedge(embedding, reverse(edge))],))
    end
    corners = messagecache(embedding_edges) do edge
        return ones(elt, (bonds[edge], bonds[nextedge(embedding, edge)]))
    end

    return CTMEnvironment(
        embedding, edgetensors, corners,
        Dict{eltype(embedding.faces), Tuple{AbstractMatrix, AbstractMatrix}}()
    )
end

# A corner `c[w => v]` touches `w` and the far end of `nextedge(w => v)`.
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
                edge in incoming if dst(nextedge(embedding, edge)) ∉ exclude
        ]
    ]
end

function environment_tensors(env::CTMEnvironment, edge::Union{AbstractEdge, Pair})
    embedding = env.embedding
    forward = NamedEdge(edge)
    backward = reverse(forward)
    return [
        edgetensor(env, forward), edgetensor(env, backward),
        corner(env, forward), corner(env, prevedge(embedding, forward)),
        corner(env, backward), corner(env, prevedge(embedding, backward)),
    ]
end

function environment_tensors(env::CTMEnvironment, face::NamedFace)
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
