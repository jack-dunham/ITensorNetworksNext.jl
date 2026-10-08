module ITensorNetworksNextTensorOperationsExt

using ITensorBase: ITensor, inds, mulopadd!, names, rename, unnamed
using ITensorNetworksNext: ITensorNetworksNext
using TensorAlgebra: TensorOperationsContract
using TensorOperations: TensorOperations as TO

# The allocator `alg` contracts with, which the intermediates must also come from.
function contract_allocator(alg::TensorOperationsContract)
    return something(alg.allocator, TO.DefaultAllocator())
end

# As in `TO.ncon`, each intermediate is an allocator temporary, freed once the next step reads it.
function ITensorNetworksNext.prod_tensors!(
        alg::TensorOperationsContract, y, x, xs...; conjlist = falses(length(xs) + 1)
    )
    allocator = contract_allocator(alg)
    checkpoint = TO.allocator_checkpoint!(allocator)
    op(i) = conjlist[i] ? conj : identity
    opx = op(1)
    for (i, m) in enumerate(Base.front(xs))
        labels = Tuple(symdiff(names(x), names(m)))
        pA, pB, pAB = TO.contract_indices(Tuple(names(x)), Tuple(names(m)), labels)
        z = TO.tensoralloc_contract(
            TO.promote_contract(eltype(x), eltype(m)), unnamed(x), pA, opx === conj,
            unnamed(m), pB, conjlist[i + 1], pAB, Val(true), allocator
        )
        z = ITensor(z, labels)
        mulopadd!(z, opx, x, op(i + 1), m, true, false; alg)
        i > 1 && TO.tensorfree!(unnamed(x), allocator)
        x, opx = z, identity
    end
    mulopadd!(y, opx, x, op(length(xs) + 1), last(xs), true, false; alg)
    length(xs) > 1 && TO.tensorfree!(unnamed(x), allocator)
    TO.allocator_reset!(allocator, checkpoint)
    return y
end

# Dense square matrices, both legs on one axis, reuse `t`'s memory and one allocator scratch
# tensor, freed on return; anything else takes the generic method.
function ITensorNetworksNext.absorb_matrices!(
        alg::TensorOperationsContract, t, matrices...
    )
    isempty(matrices) && return t
    dense = unnamed(t) isa DenseArray
    square = all(m -> eltype(m) === eltype(t) && allequal(unnamed.(inds(m))), matrices)
    if !dense || !square
        return invoke(
            ITensorNetworksNext.absorb_matrices!, Tuple{Any, Any, Vararg{Any}},
            alg, t, matrices...
        )
    end

    allocator = contract_allocator(alg)
    checkpoint = TO.allocator_checkpoint!(allocator)
    p = (ntuple(identity, ndims(unnamed(t))), ())
    scratch = TO.tensoralloc_add(eltype(t), unnamed(t), p, false, Val(true), allocator)
    x, other = t, ITensor(scratch, names(t))
    for m in matrices
        shared, added = intersect(names(m), names(x)), setdiff(names(m), names(x))
        labels = replace(names(x), only(shared) => only(added))
        y = rename(other, (names(other) .=> labels)...)
        mulopadd!(y, identity, x, identity, m, true, false; alg)
        x, other = y, x
    end
    if isodd(length(matrices))
        x = copyto!(rename(t, (names(t) .=> names(x))...), x)
    end
    TO.tensorfree!(scratch, allocator)
    TO.allocator_reset!(allocator, checkpoint)
    return x
end

end
