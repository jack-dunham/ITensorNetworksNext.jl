using AbstractTrees: nodevalue
using Combinatorics: combinations
using ITensorBase: ITensor, findname, inds, mulopadd!, name, names, rename, state, unnamed

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

"""
    absorb_matrices!(alg, t, matrices...) -> t * matrices[1] * matrices[2] * …

Contract each matrix into `t` in turn. `t` may be overwritten: a method for `alg` can reuse
its memory.
"""
absorb_matrices!(_alg, t, matrices...) = foldl(*, matrices; init = t)

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

"""
    absorb_matrices!(alg::Sliced, t, matrices...)

Form the product in `t`'s memory one block at a time, with the longest index of `t` that no
matrix carries split into at most `alg.nblocks` ranges. Each block of `t` is contracted with
all the matrices by `prod_tensors!` with `alg.contract_alg` into a block-sized scratch array
and copied back, so the matrices must be square and no wider in element type than `t`, and
each must act on a different index of `t`. If every index of `t` is carried by a matrix,
`absorb_matrices!` with `alg.contract_alg` is used.
"""
function absorb_matrices!(alg::Sliced, t, matrices...)
    isempty(matrices) && return t
    free = filter(i -> all(m -> name(i) ∉ names(m), matrices), inds(t))
    isempty(free) && return absorb_matrices!(alg.contract_alg, t, matrices...)
    for m in matrices
        if !allequal(size(m)) || promote_type(eltype(t), eltype(m)) !== eltype(t)
            throw(
                ArgumentError(
                    "`absorb_matrices!` with `Sliced` needs square matrices no wider in " *
                        "element type than $(eltype(t)), got $(eltype(m)) of size " *
                        "$(size(m))."
                )
            )
        end
    end
    replacements = map(matrices) do m
        return only(intersect(names(m), names(t))) => only(setdiff(names(m), names(t)))
    end
    t′ = rename(t, replacements...)
    i = argmax(length, free)
    step = cld(length(i), alg.nblocks)
    scratch = similar(view(t′, name(i) => 1:step))
    for r in Iterators.partition(1:length(i), step)
        y = view(scratch, name(i) => 1:length(r))
        prod_tensors!(alg.contract_alg, y, view(t, name(i) => r), matrices...)
        # The named `copyto!` aligns its source with `permutedims`, which copies even when
        # the indices are already in the destination's order, as they are here.
        copyto!(unnamed(view(t′, name(i) => r)), unnamed(y))
    end
    return t′
end
