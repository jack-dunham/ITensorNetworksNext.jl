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
    # The face to the left of each directed edge, or `nothing` for the outer face.
    leftface::Dictionary{NamedEdge{V}, Union{Nothing, NamedFace{V}}}
    # The next and previous directed edge around the face to the left of each directed edge.
    next::Dictionary{NamedEdge{V}, NamedEdge{V}}
    prev::Dictionary{NamedEdge{V}, NamedEdge{V}}
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
    leftface = Dictionary{NamedEdge{V}, Union{Nothing, NamedFace{V}}}()
    next_edges = Dictionary{NamedEdge{V}, NamedEdge{V}}()
    prev_edges = Dictionary{NamedEdge{V}, NamedEdge{V}}()
    nouter = 0
    for undirected in edges(graph),
            start in (NamedEdge{V}(undirected), reverse(NamedEdge{V}(undirected)))

        haskey(leftface, start) && continue

        cycle = [start]
        while (next = _next_edge(rotation, last(cycle))) != start
            push!(cycle, next)
        end
        for (edge, following) in zip(cycle, circshift(cycle, -1))
            set!(next_edges, edge, following)
            set!(prev_edges, following, edge)
        end

        if _signed_area(position, cycle) < 0
            nouter += 1
            foreach(edge -> set!(leftface, edge, nothing), cycle)
        else
            face = NamedFace(cycle)
            push!(faces, face)
            foreach(edge -> set!(leftface, edge, face), cycle)
        end
    end

    nouter == 1 || throw(
        ArgumentError(
            "`position` gives $nouter outer faces; a connected planar drawing has one."
        )
    )

    return PlanarEmbedding{V, typeof(graph)}(
        graph, rotation, faces, leftface, next_edges, prev_edges
    )
end

next_edge(embedding::PlanarEmbedding, edge) = embedding.next[NamedEdge(edge)]
prev_edge(embedding::PlanarEmbedding, edge) = embedding.prev[NamedEdge(edge)]
leftface(embedding::PlanarEmbedding, edge) = embedding.leftface[NamedEdge(edge)]

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
