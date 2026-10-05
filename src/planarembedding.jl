using Dictionaries: Dictionary, set!
using Graphs: AbstractGraph, dst, edges, neighbors, src, vertices
using NamedGraphs: NamedEdge

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
    rotation::Dictionary{V, Vector{V}}
    faces::Vector{NamedFace{V}}
    outer_face::NamedFace{V}
    # The face to the left of each directed edge, inner or outer, and the edge's position in it.
    positions::Dictionary{NamedEdge{V}, Tuple{NamedFace{V}, Int}}
end

function _next_edge(rotation, edge)
    source, target = src(edge), dst(edge)

    around = rotation[target]
    position = findfirst(==(source), around)
    return NamedEdge(target => around[mod1(position - 1, length(around))])
end

function _signed_area(position, cycle)
    return sum(cycle) do edge
        (x1, y1), (x2, y2) = position(src(edge)), position(dst(edge))
        return x1 * y2 - x2 * y1
    end / 2
end

"""
    planar_embedding(graph, position) -> PlanarEmbedding

Embed `graph` using the straight-line drawing that places vertex `v` at `position(v)`. The drawing
must be connected and have no crossing edges; this is not checked.
"""
function planar_embedding(graph::AbstractGraph, position)
    V = eltype(vertices(graph))
    rotation = Dictionary{V, Vector{V}}()
    for vertex in vertices(graph)
        x, y = position(vertex)
        around = collect(neighbors(graph, vertex))
        sort!(
            around;
            by = neighbor -> atan(position(neighbor)[2] - y, position(neighbor)[1] - x)
        )
        set!(rotation, vertex, around)
    end

    faces = NamedFace{V}[]
    outer_faces = NamedFace{V}[]
    positions = Dictionary{NamedEdge{V}, Tuple{NamedFace{V}, Int}}()
    for undirected in edges(graph),
            start in (NamedEdge{V}(undirected), reverse(NamedEdge{V}(undirected)))

        haskey(positions, start) && continue

        cycle = [start]
        while (next = _next_edge(rotation, last(cycle))) != start
            push!(cycle, next)
        end
        face = NamedFace(cycle)
        for (index, edge) in enumerate(cycle)
            set!(positions, edge, (face, index))
        end
        push!(_signed_area(position, cycle) < 0 ? outer_faces : faces, face)
    end

    length(outer_faces) == 1 || throw(
        ArgumentError(
            "`position` gives $(length(outer_faces)) outer faces; a connected planar drawing has one."
        )
    )

    return PlanarEmbedding{V, typeof(graph)}(
        graph, rotation, faces, only(outer_faces), positions
    )
end

function next_edge(embedding::PlanarEmbedding, edge)
    face, index = embedding.positions[NamedEdge(edge)]
    return face[mod1(index + 1, length(face))]
end
function prev_edge(embedding::PlanarEmbedding, edge)
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
