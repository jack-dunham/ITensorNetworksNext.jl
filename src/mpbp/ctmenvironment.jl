using DiagonalArrays: diagview, setdiagindices!
using Dictionaries: Dictionary
using Graphs: dst, edges, neighbors, vertices
using ITensorBase: AbstractNamedTensor, Index, IndexName, name, setname, unnamed
using NamedGraphs: NamedEdge, all_edges
using UUIDs: UUID, uuid5

struct CTMEnvironment{V, E <: MessageCache, C <: MessageCache, I}
    embedding::PlanarEmbedding{V}
    edgetensors::E
    cornertensors::C
    # Each face's `(right_basis, left_basis)` from its last update, which the next one aligns
    # onto.
    bases::Dict{NamedFace{V}, Tuple{AbstractNamedTensor, AbstractNamedTensor}}
    # The bond of each directed edge, which its corner shares with the corner before it. Kept
    # here so that a bond can be read without the corner at the edge's far end.
    envinds::Dictionary{NamedEdge{V}, I}
end

edgetensor(env::CTMEnvironment, edge) = env.edgetensors[NamedEdge(edge)]
cornertensor(env::CTMEnvironment, edge) = env.cornertensors[NamedEdge(edge)]
# The environment's indices along `edge`: those `c[edge]` shares with the corner before it.
envinds(env::CTMEnvironment, edge) = (env.envinds[NamedEdge(edge)],)

# Bond names are hashed from the edge or from the bond they replace, so that every process
# building the same environment gives each bond the same name.
const BOND_NAMESPACE = UUID("4f231e03-f233-4684-80a2-63f028edc951")
bond_index(id::UUID, n) = setname(Index(n), IndexName(; uuid = id))
initial_bond(edge) = bond_index(uuid5(BOND_NAMESPACE, string(edge)), 1)
next_bond(bond, n) = bond_index(uuid5(name(bond).uuid, "next"), n)

# Two-index tensor over `row` and `column` with `diagonal` on its diagonal; every corner is one.
function diagonal_tensor(diagonal, row, column)
    tensor = zeros(eltype(diagonal), (row, column))
    setdiagindices!(unnamed(tensor), diagonal, :)
    return tensor
end

corner_diagonal(env::CTMEnvironment, edge) = diagview(unnamed(cornertensor(env, edge)))

function Base.copy(env::CTMEnvironment)
    return CTMEnvironment(
        env.embedding, map(identity, env.edgetensors), map(identity, env.cornertensors),
        copy(env.bases), copy(env.envinds)
    )
end

"""
    ctm_environment(tn, embedding, messages) -> CTMEnvironment

The χ = 1 environment whose edge tensors are the BP `messages` and whose corners are all 1.
"""
function ctm_environment(tn, embedding::PlanarEmbedding, messages)
    embedding_edges = collect(all_edges(embedding.graph))
    envinds = Dictionary(embedding_edges, initial_bond.(embedding_edges))
    elt = eltype(messages[first(embedding_edges)])

    edgetensors = messagecache(embedding_edges) do edge
        return messages[edge] * ones(elt, (envinds[nextedge(embedding, edge)],)) *
            ones(elt, (envinds[prevedge(embedding, reverse(edge))],))
    end
    corners = messagecache(embedding_edges) do edge
        return ones(elt, (envinds[edge], envinds[nextedge(embedding, edge)]))
    end

    return CTMEnvironment(
        embedding, edgetensors, corners,
        Dict{eltype(embedding.faces), Tuple{AbstractNamedTensor, AbstractNamedTensor}}(),
        envinds
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
            cornertensor(env, edge) for
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
        cornertensor(env, forward), cornertensor(env, prevedge(embedding, forward)),
        cornertensor(env, backward), cornertensor(env, prevedge(embedding, backward)),
    ]
end

function environment_tensors(env::CTMEnvironment, face::NamedFace)
    return view(env.cornertensors, face)
end

function kikuchi_terms(tn, env::CTMEnvironment)
    graph = env.embedding.graph
    numerator = (
        vertex_scalars(tn, env, collect(vertices(graph))),
        [face_scalar(tn, env, face) for face in env.embedding.faces],
    )
    return numerator, edge_scalars(tn, env, collect(edges(graph)))
end
