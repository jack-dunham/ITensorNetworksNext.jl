using Dictionaries: Dictionary, set!
using Graphs: AbstractGraph, dst, neighbors, src, vertices
using NamedGraphs: NamedEdge, all_edges

"""
    NamedFace(edges)

A face of a planar embedding as its counterclockwise cycle of directed `edges`. Each directed
edge lies on exactly one face of an embedding, so two faces are equal when they start at the
same edge; faces from different embeddings are not comparable.
"""
struct NamedFace{V} <: AbstractVector{NamedEdge{V}}
    edges::Vector{NamedEdge{V}}
end

Base.size(face::NamedFace) = size(face.edges)
Base.getindex(face::NamedFace, i::Int) = face.edges[i]
Base.:(==)(face1::NamedFace, face2::NamedFace) = first(face1.edges) == first(face2.edges)
Base.isequal(face1::NamedFace, face2::NamedFace) = face1 == face2
Base.hash(face::NamedFace, h::UInt) = hash(first(face.edges), hash(:NamedFace, h))

struct PlanarEmbedding{V, G <: AbstractGraph}
    graph::G
    faces::Vector{NamedFace{V}}
    outer_face::NamedFace{V}
    # The face to the left of each directed edge, inner or outer, and the edge's position in it.
    positions::Dictionary{NamedEdge{V}, Tuple{NamedFace{V}, Int}}
end

function _nextedge(rotation, edge)
    source, target = src(edge), dst(edge)

    around = rotation[target]
    position = findfirst(==(source), around)
    return NamedEdge(target => around[mod1(position - 1, length(around))])
end

# The directed edges around the face to the left of `start`, beginning with `start`.
function _face_cycle(rotation, start)
    cycle = [start]
    while (next = _nextedge(rotation, last(cycle))) != start
        push!(cycle, next)
    end
    return cycle
end

function _signed_area(position, cycle)
    return sum(cycle) do edge
        (x1, y1), (x2, y2) = position(src(edge)), position(dst(edge))
        return x1 * y2 - x2 * y1
    end / 2
end

function _angle(position, vertex, neighbor)
    (x1, y1), (x2, y2) = position(vertex), position(neighbor)
    return atan(y2 - y1, x2 - x1)
end

"""
    planar_embedding(graph, position) -> PlanarEmbedding

Embed `graph` using the straight-line drawing that places vertex `v` at `position(v)`. The drawing
must be connected and have no crossing edges; this is not checked.
"""
function planar_embedding(graph::AbstractGraph, position)
    V = eltype(vertices(graph))

    # The neighbours of each vertex in counterclockwise order.
    rotation = map(vertices(graph)) do vertex
        return sort(
            neighbors(graph, vertex);
            by = neighbor -> _angle(position, vertex, neighbor)
        )
    end

    faces = NamedFace{V}[]
    positions = Dictionary{NamedEdge{V}, Tuple{NamedFace{V}, Int}}()

    outer_face = nothing

    for start in all_edges(graph)
        haskey(positions, start) && continue

        face = NamedFace(_face_cycle(rotation, start))

        for (index, edge) in enumerate(face)
            set!(positions, edge, (face, index))
        end

        # Inner faces are traced counterclockwise and the outer face clockwise.
        if _signed_area(position, face) < 0
            isnothing(outer_face) || throw(
                ArgumentError(
                    "`position` gives more than one outer face; a connected planar drawing has one."
                )
            )
            outer_face = face
        else
            push!(faces, face)
        end
    end

    isnothing(outer_face) && throw(
        ArgumentError(
            "`position` gives no outer face; a connected planar drawing has one."
        )
    )

    return PlanarEmbedding{V, typeof(graph)}(graph, faces, outer_face, positions)
end

function nextedge(embedding::PlanarEmbedding, edge)
    face, index = embedding.positions[NamedEdge(edge)]
    return face[mod1(index + 1, length(face))]
end
function prevedge(embedding::PlanarEmbedding, edge)
    face, index = embedding.positions[NamedEdge(edge)]
    return face[mod1(index - 1, length(face))]
end
# `nothing` for the outer face.
function leftface(embedding::PlanarEmbedding, edge)
    face = first(embedding.positions[NamedEdge(edge)])
    return face == embedding.outer_face ? nothing : face
end

"""
    face_coloring(embedding) -> Dictionary

A colour for each inner face of `embedding`, from `1`, such that faces sharing a vertex differ.
Faces of one colour neither read nor write each other's tensors in a face update, so updating
the colours in turn, with all faces of a colour at once, gives a sweep in colour order.
"""
function face_coloring(embedding::PlanarEmbedding)
    vertex_sets = [Set(src.(face)) for face in embedding.faces]
    colors = Int[]
    for (i, vertex_set) in enumerate(vertex_sets)
        used = Set(colors[j] for j in 1:(i - 1) if !isdisjoint(vertex_set, vertex_sets[j]))
        push!(colors, first(color for color in Iterators.countfrom(1) if color ∉ used))
    end
    return Dictionary(embedding.faces, colors)
end

# Coordinates matching NetworkX `hexagonal_lattice_graph`, whose node `(i, j)` is `(j + 1, i + 1)` here.
function hexagonal_position((row, column))
    x_index, y_index = column - 1, row - 1
    return (
        0.5 + x_index + x_index ÷ 2 + (y_index % 2) * ((x_index % 2) - 0.5),
        √3 * y_index / 2,
    )
end
