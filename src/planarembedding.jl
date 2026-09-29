using Dictionaries: Dictionary, set!
using Graphs: AbstractGraph, dst, edges, neighbors, src, vertices
using NamedGraphs: NamedEdge

struct PlanarEmbedding{V, G <: AbstractGraph}
    graph::G
    rotation::Dictionary{V, Vector{V}}
    faces::Vector{Vector{NamedEdge{V}}}
    leftface::Dictionary{NamedEdge{V}, Int}
end

function _next_dart(rotation, d)
    u, v = src(d), dst(d)
    ws = rotation[v]
    return NamedEdge(v => ws[mod1(findfirst(==(u), ws) - 1, length(ws))])
end

function _prev_dart(rotation, d)
    v, w = src(d), dst(d)
    ws = rotation[v]
    return NamedEdge(ws[mod1(findfirst(==(w), ws) + 1, length(ws))] => v)
end

function _signed_area(position, cycle)
    return sum(cycle) do d
        (x1, y1), (x2, y2) = position(src(d)), position(dst(d))
        return x1 * y2 - x2 * y1
    end / 2
end

"""
    planar_embedding(g, position) -> PlanarEmbedding

Embed `g` using the straight-line drawing that places vertex `v` at `position(v)`. The drawing
must be connected and have no crossing edges; this is not checked.
"""
function planar_embedding(g::AbstractGraph, position)
    V = eltype(vertices(g))
    rotation = Dictionary{V, Vector{V}}()
    for v in vertices(g)
        x, y = position(v)
        ws = collect(neighbors(g, v))
        sort!(ws; by = w -> atan(position(w)[2] - y, position(w)[1] - x))
        set!(rotation, v, ws)
    end
    faces = Vector{NamedEdge{V}}[]
    leftface = Dictionary{NamedEdge{V}, Int}()
    nouter = 0
    for e in edges(g), d in (NamedEdge{V}(e), reverse(NamedEdge{V}(e)))
        haskey(leftface, d) && continue
        cycle = [d]
        while (d′ = _next_dart(rotation, last(cycle))) != d
            push!(cycle, d′)
        end
        if _signed_area(position, cycle) < 0
            nouter += 1
            foreach(x -> set!(leftface, x, 0), cycle)
        else
            push!(faces, cycle)
            foreach(x -> set!(leftface, x, length(faces)), cycle)
        end
    end
    nouter == 1 || throw(
        ArgumentError(
            "`position` gives $nouter outer faces; a connected planar drawing has one."
        )
    )
    return PlanarEmbedding{V, typeof(g)}(g, rotation, faces, leftface)
end

next_dart(emb::PlanarEmbedding, d) = _next_dart(emb.rotation, NamedEdge(d))
prev_dart(emb::PlanarEmbedding, d) = _prev_dart(emb.rotation, NamedEdge(d))
leftface(emb::PlanarEmbedding, d) = emb.leftface[NamedEdge(d)]
darts(emb::PlanarEmbedding) = collect(keys(emb.leftface))

# Coordinates matching NetworkX `hexagonal_lattice_graph`, whose node `(i, j)` is `(j + 1, i + 1)` here.
function hexagonal_position((j, i))
    i0, j0 = i - 1, j - 1
    return (0.5 + i0 + i0 ÷ 2 + (j0 % 2) * ((i0 % 2) - 0.5), √3 * j0 / 2)
end
