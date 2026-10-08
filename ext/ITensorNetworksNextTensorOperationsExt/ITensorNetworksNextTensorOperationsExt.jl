module ITensorNetworksNextTensorOperationsExt

using ITensorBase: ITensor, mulopadd!, names, unnamed
using ITensorNetworksNext: ITensorNetworksNext, BlockedMessageUpdate, default_nblocks
using TensorAlgebra: TensorOperationsContract
using TensorOperations: TensorOperations as TO

# The allocator `alg` contracts with, which the intermediates must also come from.
function contract_allocator(alg::TensorOperationsContract)
    return something(alg.allocator, TO.DefaultAllocator())
end

function contract_backend(alg::TensorOperationsContract, a)
    return @something alg.backend TO.select_backend(TO.tensorcontract!, a, a, a)
end

function ITensorNetworksNext.default_nblocks(
        algorithm::BlockedMessageUpdate{<:TensorOperationsContract}, ket::AbstractArray,
        χ::Integer
    )
    return @something algorithm.nblocks default_nblocks(
        contract_backend(algorithm.contract_alg, ket),
        length(ket) * sizeof(eltype(ket)), χ
    )
end
# Splitting a leg saves memory at a cost in time, so only the device backend splits by default.
ITensorNetworksNext.default_nblocks(backend, ketbytes::Integer, χ::Integer) = 1
# About 1/16 of the ket per block, but at least 4 MiB each and at most 64 columns each.
function ITensorNetworksNext.default_nblocks(
        ::TO.cuTENSORBackend, ketbytes::Integer, χ::Integer
    )
    return max(clamp(fld(ketbytes, 4 * 2^20), 1, 16), cld(χ, 64))
end

# cuTENSOR throws `KeyError` when a contraction mixes element types, so that is rejected here
# rather than partway through a message.
function ITensorNetworksNext.check_element_types(alg::TensorOperationsContract, tensors)
    backend = contract_backend(alg, unnamed(first(tensors)))
    if backend isa TO.cuTENSORBackend && !allequal(eltype, tensors)
        throw(
            ArgumentError(
                "`BlockedMessageUpdate` on cuTENSOR needs the ket, incoming messages and " *
                    "operator to share an element type, got $(unique(map(eltype, tensors)))."
            )
        )
    end
    return nothing
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

end
