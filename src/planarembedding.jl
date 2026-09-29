using Dictionaries: Dictionary, set!
using Graphs: AbstractGraph, dst, edges, neighbors, src, vertices
using NamedGraphs: NamedEdge

struct PlanarEmbedding{V, G <: AbstractGraph}
    graph::G
    rotation::Dictionary{V, Vector{V}}
    faces::Vector{Vector{NamedEdge{V}}}
    leftface::Dictionary{NamedEdge{V}, Int}
end

function _next_edge(rotation, edge)
    source, target = src(edge), dst(edge)

    around = rotation[target]
    position = findfirst(==(source), around)
    return NamedEdge(target => around[mod1(position - 1, length(around))])
end

function _prev_edge(rotation, edge)
    source, target = src(edge), dst(edge)

    around = rotation[source]
    position = findfirst(==(target), around)
    return NamedEdge(around[mod1(position + 1, length(around))] => source)
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

    faces = Vector{NamedEdge{V}}[]
    leftface = Dictionary{NamedEdge{V}, Int}()
    nouter = 0
    for undirected in edges(graph),
            start in (NamedEdge{V}(undirected), reverse(NamedEdge{V}(undirected)))

        haskey(leftface, start) && continue

        cycle = [start]
        while (next = _next_edge(rotation, last(cycle))) != start
            push!(cycle, next)
        end

        if _signed_area(position, cycle) < 0
            nouter += 1
            foreach(edge -> set!(leftface, edge, 0), cycle)
        else
            push!(faces, cycle)
            foreach(edge -> set!(leftface, edge, length(faces)), cycle)
        end
    end

    nouter == 1 || throw(
        ArgumentError(
            "`position` gives $nouter outer faces; a connected planar drawing has one."
        )
    )

    return PlanarEmbedding{V, typeof(graph)}(graph, rotation, faces, leftface)
end

function next_edge(embedding::PlanarEmbedding, edge)
    return _next_edge(embedding.rotation, NamedEdge(edge))
end
function prev_edge(embedding::PlanarEmbedding, edge)
    return _prev_edge(embedding.rotation, NamedEdge(edge))
end
leftface(embedding::PlanarEmbedding, edge) = embedding.leftface[NamedEdge(edge)]

# Coordinates matching NetworkX `hexagonal_lattice_graph`, whose node `(i, j)` is `(j + 1, i + 1)` here.
function hexagonal_position((row, column))
    x_index, y_index = column - 1, row - 1
    return (
        0.5 + x_index + x_index ÷ 2 + (y_index % 2) * ((x_index % 2) - 0.5),
        √3 * y_index / 2,
    )
end
