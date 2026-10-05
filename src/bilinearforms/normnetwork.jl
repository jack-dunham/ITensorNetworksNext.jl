using Dictionaries: Dictionary, dictionary, set!
using ITensorBase: inputnames, outputnames, similar_operator, uniquename
using ITensorNetworksNext

"""
    struct NormNetwork{T, V, I} <: AbstractBilinearFormNetwork{T, V, I}

Lazy wrapper representing the norm `⟨tn|tn⟩` of `tn::ITensorNetwork{T, V, I}`,
together with a per-edge ket→bra name mapping that, for each index in the ket layer, defines
the name of the corresponding index in the bra layer.

    NormNetwork(ket, map, [names])

The bra layer takes `map[name]` for each link name of `ket` and each name in `names`; every
other name is shared by both layers.
"""
struct NormNetwork{T, V, I} <: AbstractBilinearFormNetwork{T, V, I}
    ket::ITensorNetwork{T, V, I}
    braname::Dictionary{I, I}
    function NormNetwork(
            ket::ITensorNetwork{T, V, I},
            map::Dictionary{I, I},
            names = ()
        ) where {T, V, I}
        return new{T, V, I}(ket, select_branames(ket, map, names))
    end
end

function Base.eltype(::Type{<:NormNetwork})
    return error(
        "`eltype` of a `NormNetwork` is not defined, since the double-layer tensor at a vertex has no representation of its own. Use `kettensor` and `bratensor` to reach the individual layers."
    )
end

function NormNetwork(tn::ITensorNetwork)
    return NormNetwork(tn, map(uniquename, keys(tn.dimname_vertices)))
end

# ==================================== DataGraphs.jl ===================================== #

function DataGraphs.get_vertex_data(nn::NormNetwork, vertex)
    return error(
        "Indexing a `NormNetwork` is not defined, since the double-layer tensor at a vertex has no representation of its own. Use `kettensor` and `bratensor` to reach the individual layers."
    )
end

# ====================================== interface ======================================= #

ketnetwork(nn::NormNetwork) = nn.ket
branamemap(nn::NormNetwork) = nn.braname

"""
    flatten_network(nn::NormNetwork) -> ITensorNetwork

Expand a norm network into a plain tensor network carrying one vertex per layer, so that vertex
`v` of `nn` becomes the two vertices `(v, :ket)` and `(v, :bra)` and `nv` doubles. A vertex's
two layers stay adjacent in the vertex order, ket first.
"""
function flatten_network(nn::NormNetwork)
    return ITensorNetwork(
        dictionary(
            Iterators.flatten(
                ((v, :ket) => kettensor(nn, v), (v, :bra) => bratensor(nn, v))
                    for v in vertices(nn)
            )
        )
    )
end

"""
    normnetwork(tn::ITensorNetwork, [braname]) -> NormNetwork

Build the double-layer norm network `⟨tn|tn⟩`, represented lazily as a `NomnNetwork` object.
The optional second argument `braname` should implement `braname[ketdimname] = bradimname` for
every link dimension name `ketdimname` in `tn`. If this is not specified, then a name is
generated via the `ITensorBase.uniquename` function.
"""
normnetwork(tn::ITensorNetwork) = NormNetwork(tn)
normnetwork(tn::ITensorNetwork, braname) = NormNetwork(tn, braname)

"""
    message_normnetwork(tn::ITensorNetwork, messages) -> NormNetwork

The norm network of `tn` whose bra layer uses the names of the operator-valued `messages`:
each input name of a message (a ket bond) maps to the output name at the same position. A
bond held by a single vertex of `tn` is mapped too, provided a message carries it.
"""
function message_normnetwork(tn::ITensorNetwork{T, V, I}, messages) where {T, V, I}
    braname = Dictionary{I, I}()
    for edge in edges(messages)
        message = messages[edge]
        for (ketname, braname_edge) in zip(inputnames(message), outputnames(message))
            set!(braname, ketname, braname_edge)
        end
    end
    return NormNetwork(tn, braname, keys(braname))
end
