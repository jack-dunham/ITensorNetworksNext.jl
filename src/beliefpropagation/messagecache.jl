using DataGraphs: DataGraphs, AbstractDataGraph, AbstractEdgeDataGraph, edge_data,
    edge_data_type, set_vertex_data!, underlying_graph, underlying_graph_type, vertex_data,
    vertex_data_type
using Dictionaries: Dictionary, Indices, set!, unset!
using Graphs: AbstractGraph, connected_components, edges, is_directed, is_tree
using ITensorBase: inputnames, outputnames, state, unnamed
using NamedGraphs: AbstractNamedEdge, NamedDiGraph, NamedEdge, add_edges!, arrange_edge,
    edge_subgraph, in_incident_edges, incident_edges, to_graph_index, vertextype
using SplitApplyCombine: mapmany

struct MessageCache{T, V} <: AbstractEdgeDataGraph{T, V}
    messages::Dictionary{NamedEdge{V}, T}
    underlying_graph::NamedDiGraph{V}
    function MessageCache{T, V}(::UndefInitializer, vertices) where {T, V}
        messages = Dictionary{NamedEdge{V}, T}()
        underlying_graph = NamedDiGraph{V}(vertices)
        return new{T, V}(messages, underlying_graph)
    end
end

# single type parameter version of the inner constructor
function MessageCache(::UndefInitializer, vertices)
    return MessageCache{Any}(undef, vertices)
end
function MessageCache{T}(::UndefInitializer, vertices) where {T}
    return MessageCache{T, eltype(vertices)}(undef, vertices)
end

MessageCache(messages) = MessageCache{valtype(messages)}(messages)

function MessageCache{T}(messages) where {T}
    V = vertextype(keytype(messages))
    return MessageCache{T, V}(messages)
end

# `messages` is any iterable data structure, where `keys(messages)` are edges
# and the values are the messages on those edges.
function MessageCache{T, V}(messages) where {T, V}
    edges = keys(messages)
    vertices = union(src.(edges), dst.(edges))
    cache = MessageCache{T, V}(undef, vertices)
    add_edges!(cache.underlying_graph, edges)
    copyto!(cache, messages)
    return cache
end

messagecache(pairs) = MessageCache(Dict(pairs))
messagecache(f, edges) = messagecache(edge => f(edge) for edge in edges)

function Graphs.rem_edge!(c::MessageCache, edge)
    unset!(c.messages, to_graph_index(c, edge))
    return rem_edge!(c.underlying_graph, edge)
end

function Graphs.add_vertex!(c::MessageCache, vertex)
    return add_vertex!(c.underlying_graph, vertex)
end

function Graphs.has_edge(c::MessageCache, edge::AbstractNamedEdge)
    return has_edge(c.underlying_graph, edge)
end

# ================================ NamedGraphs interface ================================= #

function NamedGraphs.similar_graph(::Type{<:MessageCache}, vertices)
    return MessageCache(undef, vertices)
end

function NamedGraphs.similar_graph(cache::MessageCache, T::Type)
    new_cache = similar_graph(cache, T, vertices(cache))
    add_edges!(new_cache.underlying_graph, edges(cache))
    return new_cache
end
function NamedGraphs.similar_graph(::MessageCache, ED::Type, vertices)
    return MessageCache{ED}(undef, collect(vertices))
end

# ================================= DataGraphs interface ================================= #

DataGraphs.underlying_graph(cache::MessageCache) = cache.underlying_graph

DataGraphs.is_vertex_assigned(::MessageCache, _) = false
DataGraphs.is_edge_assigned(c::MessageCache, edge) = haskey(c.messages, edge)

DataGraphs.get_edge_data(c::MessageCache, edge::AbstractEdge) = c.messages[edge]
function DataGraphs.set_edge_data!(c::MessageCache, val, edge)
    has_edge(c, edge) || add_edge!(c.underlying_graph, edge)
    set!(c.messages, edge, val)
    return c
end

function DataGraphs.insert_edge_data!(cache::MessageCache, edge, val)
    add_edge!(cache.underlying_graph, edge)
    insert!(cache.messages, edge, val)
    return cache
end

# =================================== Dictionaries.jl ==================================== #

Dictionaries.issettable(::MessageCache) = true
Dictionaries.isinsertable(::MessageCache) = true

# The messages on `edges`, in that order, as a DataGraphs view of the cache's edge data; writes
# to an existing edge go through to the cache.
function Base.view(cache::MessageCache, edges::AbstractVector{<:AbstractEdge})
    return view(cache, Indices(edges))
end

function Base.map(f, cache::MessageCache)
    new_cache = similar_graph(cache, Base.promote_op(f, valtype(cache)))
    map!(f, new_cache, cache)
    return new_cache
end

function Base.map!(f, dst::MessageCache, src)
    for key in keys(src)
        dst[key] = f(src[key])
    end
    return dst
end

# ===================================== contraction ====================================== #

function incoming_messages(cache::AbstractGraph, pair::Pair)
    edge = to_graph_index(cache, pair)
    return incoming_messages(cache, edge)
end
function incoming_messages(cache::AbstractGraph, edge::AbstractEdge)
    return environment_tensors(cache, src(edge); exclude = (dst(edge),))
end

# A vertex is anything that is not an edge or a `Pair`. `exclude` drops the tensors that touch
# the listed neighbours of `vertex`.
function environment_tensors(messages::AbstractGraph, vertex; exclude = ())
    return [messages[e] for e in in_incident_edges(messages, vertex) if src(e) ∉ exclude]
end

"""
    incident_subgraph(graph, vertices)

The subgraph of `graph` spanned by the edges with an endpoint in `vertices`. Unlike
`subgraph`, it includes the neighbours of `vertices`, but no edge between two neighbours.
"""
function incident_subgraph(graph::AbstractGraph, vertices)
    edges = mapreduce(v -> incident_edges(graph, v; dir = :both), union, vertices)
    return edge_subgraph(graph, edges)
end

function environment_tensors(messages::AbstractGraph, edge::Union{AbstractEdge, Pair})
    return [messages[edge], messages[reverse(edge)]]
end

# `tensor`, when given, stands in for the factor tensors at `vertex`.
function vertex_scalar(factors, env, vertex; tensor = nothing, alg = Greedy())
    return vertex_scalar(alg, factors, env, vertex; tensor)
end
function vertex_scalar(alg, factors, env, vertex; tensor = nothing)
    vertex_tensors = isnothing(tensor) ? factor_tensors(factors, vertex) : [tensor]
    tensors = [vertex_tensors; environment_tensors(env, vertex)]
    return prod_tensors(tensors, alg)[]
end
# A scalar has no open index to slice, so one environment tensor is contracted last, leaving its
# link open in the sliced product.
function vertex_scalar(alg::Sliced, factors, env, vertex; tensor = nothing)
    env_tensors = environment_tensors(env, vertex)
    isempty(env_tensors) && return vertex_scalar(Greedy(), factors, env, vertex; tensor)
    rest = env_tensors[2:end]
    tensors, conjlist = if isnothing(tensor)
        message_contraction_tensors(factors, rest, vertex)
    else
        [tensor; rest], falses(length(rest) + 1)
    end
    return (prod_tensors(tensors, alg; conjlist) * first(env_tensors))[]
end

function vertex_scalars(factors, messages; kwargs...)
    return vertex_scalars(factors, messages, keys(factors); kwargs...)
end
function vertex_scalars(factors::AbstractGraph, messages; kwargs...)
    return vertex_scalars(factors, messages, vertices(factors); kwargs...)
end
# `vertex_scalar` reads a number out of an `ITensor`, whose array field is untyped, so `map` would
# give element type `Any`; collecting the values instead picks up the type they actually have.
function vertex_scalars(factors, messages, vertices; kwargs...)
    return narrow_map(v -> vertex_scalar(factors, messages, v; kwargs...), vertices)
end

# Takes factors as an unused argument for consistency with `vertex_scalar`.
edge_scalar(_factors, env, edge) = prod_tensors(environment_tensors(env, edge), Greedy())[]
face_scalar(_factors, env, face) = prod_tensors(environment_tensors(env, face), Greedy())[]
edge_scalars(factors, messages) = edge_scalars(factors, messages, edges(factors))
function edge_scalars(factors, messages, edges)
    return narrow_map(e -> edge_scalar(factors, messages, e), edges)
end

function region_scalar(factors, messages, region)
    return mapreduce(vertex -> vertex_scalar(factors, messages, vertex), *, region)
end

# (log|∏terms|, sign(∏terms))
function sumlogabs(terms)
    T = eltype(terms)

    return mapreduce(
        t -> (log(abs(t)), sign(t)),
        ((d1, s1), (d2, s2)) -> (d1 + d2, s1 * s2),
        terms; init = (zero(float(real(T))), one(T))
    )
end

function sumlog(terms)
    d, s = sumlogabs(terms)
    return s isa Real && s > 0 ? d : d + log(complex(s))
end

# Returns a tuple of numerator term collections and the denominator terms.
function kikuchi_terms(factors, messages; kwargs...)
    return (vertex_scalars(factors, messages; kwargs...),), edge_scalars(factors, messages)
end

# We need a graph structure here, so assume `factors` is a graph.
function bethe_free_entropy(factors, env; kwargs...)
    numerator_terms, denominator_terms = kikuchi_terms(factors, env; kwargs...)

    if any(iszero, denominator_terms)
        return -Inf
    end

    return sum(sumlog, numerator_terms) - sumlog(denominator_terms)
end
function bethe_free_energy(factors, messages; kwargs...)
    return -bethe_free_entropy(factors, messages; kwargs...)
end

function expect(factors, env, vertex, tensor)
    return vertex_scalar(factors, env, vertex; tensor) / vertex_scalar(factors, env, vertex)
end

# ============================= AbstractBilinearFormNetwork ============================== #

# Each message maps its input (ket) names to its output (bra) names; both messages on an edge
# carry the same pair.
function branamemap(messages::MessageCache)
    pairs = [
        ket => bra for e in edges(messages) for
            (ket, bra) in zip(inputnames(messages[e]), outputnames(messages[e]))
    ]
    map = Dictionary{eltype(first.(pairs)), eltype(last.(pairs))}()
    for (ket, bra) in pairs
        set!(map, ket, bra)
    end
    return map
end

function similar_message_environment(nn::AbstractBilinearFormNetwork)
    messages = mapmany(vertices(nn)) do vertex
        return map(in_incident_edges(nn, vertex)) do edge
            bra = branetwork(nn)
            ket = ketnetwork(nn)

            ketnames = linknames(ket, edge)
            branames = linknames(bra, edge)
            braaxis = unnamed.(linkaxes(bra, edge))

            # Bra leg = operator output, ket leg = input, the bipartition in which the message
            # is positive semidefinite.
            message = similar_operator(ket[vertex], braaxis, branames, ketnames)

            return edge => message
        end
    end

    return messagecache(messages)
end

function message_environment(f::Base.Callable, nn::AbstractBilinearFormNetwork)
    return map(f, similar_message_environment(nn))
end
