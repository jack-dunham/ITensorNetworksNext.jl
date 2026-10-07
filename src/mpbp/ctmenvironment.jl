using DiagonalArrays: diagview, setdiagindices!
using Dictionaries: Dictionary
using Graphs: dst, edges, neighbors, vertices
using ITensorBase: AbstractNamedTensor, Index, NamedOneTo, unnamed
using NamedGraphs: NamedEdge, all_edges

struct CTMEnvironment{V, E <: MessageCache, C <: MessageCache, I}
    embedding::PlanarEmbedding{V}
    edgetensors::E
    cornertensors::C
    bases::Dict{NamedFace{V}, Tuple{AbstractNamedTensor, AbstractNamedTensor}}
    envinds::Dictionary{NamedEdge{V}, I}
end

edgetensor(env::CTMEnvironment, edge) = env.edgetensors[NamedEdge(edge)]
cornertensor(env::CTMEnvironment, edge) = env.cornertensors[NamedEdge(edge)]
# The environment's indices along `edge`: those `c[edge]` shares with the corner before it.
envinds(env::CTMEnvironment, edge) = env.envinds[NamedEdge(edge)]

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
    ctm_environment(tn, embedding, messages; bondnames = nothing) -> CTMEnvironment

The χ = 1 environment whose edge tensors are the BP `messages` and whose corners are all 1.
`bondnames` maps each directed edge to the name of its bond; by default the names are new.
"""
function ctm_environment(tn, embedding::PlanarEmbedding, messages; bondnames = nothing)
    embedding_edges = collect(all_edges(embedding.graph))
    bond(edge) = isnothing(bondnames) ? Index(1) : NamedOneTo(1, bondnames[edge])
    envinds = Dictionary(embedding_edges, [(bond(edge),) for edge in embedding_edges])
    elt = eltype(messages[first(embedding_edges)])

    edgetensors = messagecache(embedding_edges) do edge
        return messages[edge] * ones(elt, envinds[nextedge(embedding, edge)]) *
            ones(elt, envinds[prevedge(embedding, reverse(edge))])
    end
    corners = messagecache(embedding_edges) do edge
        return ones(elt, (envinds[edge]..., envinds[nextedge(embedding, edge)]...))
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
