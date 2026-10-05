module ITensorNetworksNextTensorOperationsExt

using Graphs: src
using ITensorBase: ITensorBase, ITensor, inds, inputnames, mulopadd!, names, outputnames,
    rename, state, unnamed
using ITensorNetworksNext: ITensorNetworksNext, AbstractBilinearFormNetwork,
    BlockedMessageUpdate, NormGramian, QuadraticFormGramian, braname, check_element_types,
    default_nblocks, incoming_messages, kettensor, operatortensor, prod_tensors!,
    updated_message
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

# The ket and the tensors each block contracts into its slice in turn, checked before any
# contraction runs.
function message_contraction_tensors(algorithm, factor::NormGramian, messages)
    rest = map(state, collect(messages))
    return checked_tensors(algorithm, kettensor(factor), rest)
end
# The operator is one more step of the block contraction, so it may carry only its paired site
# indices: a link index to a neighbouring operator would be a third leg on the message.
function message_contraction_tensors(algorithm, factor::QuadraticFormGramian, messages)
    op = factor.operator

    linkinds = setdiff(names(op), [inputnames(op); outputnames(op)])

    if !isempty(linkinds)
        throw(
            ArgumentError(
                "`BlockedMessageUpdate` needs a product operator, but the operator has " *
                    "indices $linkinds besides its inputs and outputs."
            )
        )
    end

    rest = [operatortensor(factor); map(state, collect(messages))]

    return checked_tensors(algorithm, kettensor(factor), rest)
end

# Non-dense storage cannot be sliced by column.
function checked_tensors(algorithm, ket, rest)
    tensors = [ket; rest]
    for t in tensors
        unnamed(t) isa DenseArray || throw(
            ArgumentError(
                "`BlockedMessageUpdate` requires dense storage, got $(typeof(unnamed(t)))."
            )
        )
    end
    check_element_types(algorithm.contract_alg, tensors)
    return ket, rest
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

function ITensorNetworksNext.updated_message(
        algorithm::BlockedMessageUpdate, cache, factors::AbstractBilinearFormNetwork, edge
    )
    factor = factors[src(edge)]
    messages = incoming_messages(cache, edge)

    ket, rest = message_contraction_tensors(algorithm, factor, messages)

    # Shares the ket's data; the closing contraction conjugates it through its `conj` op.
    bra = rename(n -> braname(factor, n), ket)
    # The far vertex of `edge` may not be in `factors`, so the leg is found on the message.
    ketdimname = only(intersect(names(cache[edge]), names(ket)))

    χ = size(ket, ITensorBase.dim(ket, ketdimname))

    nblocks = min(default_nblocks(algorithm, unnamed(ket), χ), χ)

    # The message being replaced has the output's indices, bra copy then ket leg.
    T = promote_type(eltype(ket), map(eltype, rest)...)
    out = similar(ket, T, Tuple(inds(state(cache[edge]))))
    conjlist = [falses(length(rest) + 1); true]

    for block in 1:nblocks
        cols = (fld((block - 1) * χ, nblocks) + 1):fld(block * χ, nblocks)
        prod_tensors!(
            algorithm.contract_alg, view(out, ketdimname => cols),
            view(ket, ketdimname => cols), rest..., bra; conjlist
        )
    end
    return out
end

end
