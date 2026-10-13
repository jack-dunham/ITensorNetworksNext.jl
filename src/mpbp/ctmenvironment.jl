using DiagonalArrays: setdiagindices!
using Dictionaries: Dictionary
using Graphs: dst, edges, neighbors, vertices
using ITensorBase: AbstractNamedTensor, NamedUnitRange, unnamed
using NamedGraphs: NamedEdge, all_edges
using TensorAlgebra: trivialrange

# A face's corners are all `Λ^(1/m)` for its eigenvalues `Λ`, so only `Λ` is stored, by face.
struct CTMEnvironment{V, E <: MessageCache, I}
    embedding::PlanarEmbedding{V}
    edgetensors::E
    eigenvalues::Dict{NamedFace{V}, AbstractVector}
    bases::Dict{NamedFace{V}, Tuple{AbstractNamedTensor, AbstractNamedTensor}}
    envinds::Dictionary{NamedEdge{V}, I}
end

edgetensor(env::CTMEnvironment, edge) = env.edgetensors[NamedEdge(edge)]

# The corner `c[edge]`, between the bonds on `edge` and on `nextedge(edge)`. Corners of the outer
# face are 1.
function cornertensor(env::CTMEnvironment, edge)
    row = only(envinds(env, edge))
    column = only(envinds(env, nextedge(env.embedding, edge)))
    face = leftface(env.embedding, edge)
    isnothing(face) && return ones((row, column))
    return diagonal_tensor(corner_roots(env, face), row, column)
end
# The environment's indices along `edge`: those `c[edge]` shares with the corner before it.
envinds(env::CTMEnvironment, edge) = env.envinds[NamedEdge(edge)]

# Two-index tensor over `row` and `column` with `diagonal` on its diagonal; every corner is one.
function diagonal_tensor(diagonal, row, column)
    tensor = zeros(eltype(diagonal), (row, column))
    setdiagindices!(unnamed(tensor), diagonal, :)
    return tensor
end

# The diagonal of each corner of the `m`-edge `face`: the `m`-th roots of its eigenvalues, which
# keep each corner's condition number the `m`-th root of the eigenvalues'.
function corner_roots(env::CTMEnvironment, face)
    eigenvalues, m = env.eigenvalues[face], length(face)
    all(value -> isreal(value) && real(value) > 0, eigenvalues) &&
        return real.(eigenvalues) .^ (1 / m)
    return complex.(eigenvalues) .^ (1 / m)
end

function Base.copy(env::CTMEnvironment)
    return CTMEnvironment(
        env.embedding, map(identity, env.edgetensors), copy(env.eigenvalues),
        copy(env.bases), copy(env.envinds)
    )
end

"""
    ctm_environment(tn, embedding, messages; bondnames = nothing) -> CTMEnvironment

The χ = 1 environment whose edge tensors are the BP `messages` and whose eigenvalues are all 1.
`bondnames` maps each directed edge to the name of its bond; by default the names are new.
"""
function ctm_environment(
        tn, embedding::PlanarEmbedding, messages;
        bondnames = map(_ -> uniquename(nametype(tn)), Indices(all_edges(tn)))
    )
    edges = all_edges(tn)
    @assert issetequal(edges, all_edges(embedding.graph))

    # Each bond starts as the identity range for fusing the network's link ranges.
    trivial = trivialrange(unnamed(first(linkinds(tn, first(edges)))))
    bondinds = map(name -> (NamedUnitRange(trivial, name),), Dictionary(bondnames))

    T = eltype(first(messages))

    edgetensors = messagecache(edges) do edge
        return messages[edge] *
            ones(T, bondinds[nextedge(embedding, edge)]) *
            ones(T, bondinds[prevedge(embedding, reverse(edge))])
    end
    eigenvalues = Dict{eltype(embedding.faces), AbstractVector}(
        face => ones(T, 1) for face in embedding.faces
    )

    return CTMEnvironment(
        embedding, edgetensors, eigenvalues,
        Dict{eltype(embedding.faces), Tuple{AbstractNamedTensor, AbstractNamedTensor}}(),
        bondinds
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
    return [cornertensor(env, edge) for edge in face]
end

function kikuchi_terms(tn, env::CTMEnvironment)
    graph = env.embedding.graph
    numerator = (
        vertex_scalars(tn, env, collect(vertices(graph))),
        [face_scalar(tn, env, face) for face in env.embedding.faces],
    )
    return numerator, edge_scalars(tn, env, collect(edges(graph)))
end
