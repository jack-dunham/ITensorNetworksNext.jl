using AbstractTrees: nodevalue
using Combinatorics: combinations
using ITensorBase: ITensor, findname, inds, mulopadd!, name, names, state, unnamed

"""
    ContractionTreeAlgorithm

Supertype of the strategies for finding a contraction order. See [`contraction_tree`](@ref).
"""
abstract type ContractionTreeAlgorithm <: AbstractAlgorithm end

"""
    prod_tensors(tensors, tree::ContractionTree; conjlist = nothing)
    prod_tensors(tensors, alg; conjlist = nothing)
    prod_tensors(tensors)

Contract `tensors` in the pairwise order given by `tree`, whose leaf labels index into
`tensors`. Given an order algorithm instead, find a tree with it first. Given neither, contract
in key order, folding from the left. The tensors whose labels `conjlist` flags are conjugated.

`tensors` is anything indexable by those labels, so a `Vector` pairs with a tree over positions
and a tensor network pairs with a tree over vertices.
"""
function prod_tensors end
prod_tensors(tensors) = prod_tensors(tensors, left_associative_tree(keys(tensors)))
function prod_tensors(tensors, tree::ContractionTree; conjlist = nothing)
    if isleaf(tree)
        k = nodevalue(tree)
        return !isnothing(conjlist) && conjlist[k] ? conj(tensors[k]) : tensors[k]
    end
    return prod_tensors(tensors, tree[1]; conjlist) *
        prod_tensors(tensors, tree[2]; conjlist)
end
function prod_tensors(tensors, alg::ContractionTreeAlgorithm; kwargs...)
    return prod_tensors(tensors, contraction_tree(alg, tensors); kwargs...)
end

# `y = x * xs...` left to right with `alg`, conjugating the operands flagged in `conjlist`.
function prod_tensors!(alg, y, x, xs...; conjlist = falses(length(xs) + 1))
    op(i) = conjlist[i] ? conj : identity
    opx = op(1)
    for (i, m) in enumerate(Base.front(xs))
        labels = Tuple(symdiff(names(x), names(m)))
        dims = map(labels) do n
            return n in names(x) ? size(x, findname(x, n)) : size(m, findname(m, n))
        end
        T = promote_type(eltype(x), eltype(m))
        z = ITensor(similar(unnamed(x), T, dims), labels)
        mulopadd!(z, opx, x, op(i + 1), m, true, false; alg)
        x, opx = z, identity
    end
    return mulopadd!(y, opx, x, op(length(xs) + 1), last(xs), true, false; alg)
end

# The tree that folds `labels` from the left, so `(a, b, c)` gives `((a, b), c)`.
function left_associative_tree(labels)
    isempty(labels) && throw(ArgumentError("No tensors to contract."))
    return reduce(ContractionTree, map(ContractionTree, labels))
end

"""
    contraction_tree(tensors; alg = Greedy())
    contraction_tree(alg, tensors)

Find a pairwise contraction order for `tensors`, returned as a [`ContractionTree`](@ref) over
their keys.

Only the index names and lengths are consulted, never the entries, so an order found once can
be replayed against any network of the same shape.
"""
function contraction_tree end
contraction_tree(tensors; alg = Greedy()) = contraction_tree(alg, tensors)
function contraction_tree(alg, tensors)
    return throw(ArgumentError("Contraction order algorithm `$(alg)` not implemented."))
end

"""
    Greedy

Repeatedly contract the cheapest available pair, measured as the product of the lengths of all
indices involved. Outer products are taken only once nothing else is left.
"""
struct Greedy <: ContractionTreeAlgorithm end

function contraction_tree(::Greedy, tensors)
    ks = collect(keys(tensors))
    isempty(ks) && throw(ArgumentError("No tensors to contract."))
    trees = map(ContractionTree, ks)
    # The inds of each pending operand. An intermediate never materializes, so its inds are
    # tracked here rather than read off a tensor.
    is = [inds(tensors[k]) for k in ks]
    while length(trees) > 1
        i1, i2 = argmin(combinations(eachindex(trees), 2)) do (i, j)
            # Defer outer products: with nothing contracted they cost the full product of both
            # operands, and taking one early inflates every contraction that follows.
            isdisjoint(is[i], is[j]) && return typemax(Int)
            # Every index on either operand is iterated once, shared ones counted once.
            return prod(length, union(is[i], is[j]); init = 1)
        end
        # Shared indices are summed over, so the result carries the symmetric difference.
        contracted = symdiff(is[i1], is[i2])
        tree = ContractionTree(trees[i1], trees[i2])
        # Remove the pair by position. Removing it by value would also drop any other operand
        # equal to it, silently losing a tensor from a network with repeated tensors.
        keep = [i for i in eachindex(trees) if i ∉ (i1, i2)]
        trees = [trees[keep]; [tree]]
        is = [is[keep]; [contracted]]
    end
    return only(trees)
end

"""
    Sliced(; nblocks::Int, contract_alg = nothing)

Contract left to right in the order the tensors are given, one block of the result at a time,
with the result's longest index split into ranges of length `cld(length, nblocks)`, so at most
`nblocks` of them. Each block is written in place by `prod_tensors!` with `contract_alg`, which
conjugates the tensors flagged in `conjlist` within the contraction rather than copying them.
"""
@kwdef struct Sliced{ContractAlg} <: AbstractAlgorithm
    nblocks::Int
    contract_alg::ContractAlg = nothing
end

function prod_tensors(tensors, alg::Sliced; conjlist = falses(length(tensors)))
    tensors = map(state, tensors)
    isone(length(tensors)) && return only(conjlist) ? conj(only(tensors)) : only(tensors)
    is = mapreduce(inds, symdiff, tensors)
    i = argmax(length, is)
    T = mapreduce(eltype, promote_type, tensors)
    storage = similar(unnamed(first(tensors)), T, Tuple(map(length, is)))
    result = ITensor(storage, Tuple(map(name, is)))
    for r in Iterators.partition(1:length(i), cld(length(i), alg.nblocks))
        block = map(t -> name(i) in names(t) ? view(t, name(i) => r) : t, tensors)
        prod_tensors!(alg.contract_alg, view(result, name(i) => r), block...; conjlist)
    end
    return result
end
