module ITensorNetworksNextTensorOperationsExt

using Graphs: src
using ITensorBase: ITensorBase, AbstractNamedTensor, ITensor, inds, inputnames, mulopadd!,
    names, outputnames, rename, state, unnamed
using ITensorNetworksNext: ITensorNetworksNext, AbstractBilinearFormNetwork,
    BlockedMessageUpdate, NormGramian, QuadraticFormGramian, braname, default_nblocks,
    incoming_messages, kettensor, operatortensor, updated_message
using TensorAlgebra: TensorAlgebra as TA, TensorOperationsContract
using TensorOperations: TensorOperations as TO

# The allocator `alg` contracts with, which the intermediates must also come from.
function contract_allocator(alg::TensorOperationsContract)
    return something(alg.allocator, TO.DefaultAllocator())
end
contract_allocator(alg) = TO.DefaultAllocator()

function contract_backend(alg::TensorOperationsContract, a)
    return @something alg.backend TO.select_backend(TO.tensorcontract!, a, a, a)
end
contract_backend(alg, a) = nothing

function ITensorNetworksNext.default_nblocks(
        algorithm::BlockedMessageUpdate, ket::AbstractArray, χ::Integer
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

# The algorithm for the steps whose outputs are freed with `tensorfree!`, which `BufferAllocator`
# only serves from its buffer when the allocation is marked temporary.
function temporary_alg(alg::TensorOperationsContract)
    return TensorOperationsContract(alg.backend, alg.allocator, true)
end
temporary_alg(alg) = alg

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

# Non-dense storage cannot be sliced by column. cuTENSOR throws `KeyError` when a contraction
# mixes element types, so that is rejected here rather than partway through a message.
function checked_tensors(algorithm, ket, rest)
    tensors = [ket; rest]
    for t in tensors
        unnamed(t) isa DenseArray || throw(
            ArgumentError(
                "`BlockedMessageUpdate` requires dense storage, got $(typeof(unnamed(t)))."
            )
        )
    end
    backend = contract_backend(algorithm.contract_alg, unnamed(ket))
    if backend isa TO.cuTENSORBackend && !allequal(eltype, tensors)
        throw(
            ArgumentError(
                "`BlockedMessageUpdate` on cuTENSOR needs the ket, incoming messages and " *
                    "operator to share an element type, got $(unique(map(eltype, tensors)))."
            )
        )
    end
    return ket, rest
end

# `TensorAlgebra.contract` on named tensors, matching dimensions by name: `alg` selects the kernel
# and, for `TensorOperationsContract`, the allocator of the output. Named `*` takes no algorithm.
function prod_tensors(a1::AbstractNamedTensor, a2::AbstractNamedTensor; kwargs...)
    a, labels = TA.contract(unnamed(a1), names(a1), unnamed(a2), names(a2); kwargs...)
    return ITensor(a, labels)
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

    contract_alg = algorithm.contract_alg
    allocator = contract_allocator(contract_alg)
    step_alg = temporary_alg(contract_alg)

    # The message being replaced has the output's indices, bra copy then ket leg.
    T = promote_type(eltype(ket), map(eltype, rest)...)
    out = similar(ket, T, Tuple(inds(state(cache[edge]))))

    for block in 1:nblocks
        cols = (fld((block - 1) * χ, nblocks) + 1):fld(block * χ, nblocks)
        checkpoint = TO.allocator_checkpoint!(allocator)
        slice = view(ket, ketdimname => cols)
        x = slice
        for m in rest
            y = prod_tensors(x, m; alg = step_alg)
            x === slice || TO.tensorfree!(unnamed(x), allocator)
            # `x` is now bound to y, an object allocated via `allocator`.
            x = y
        end
        mulopadd!(
            view(out, ketdimname => cols), conj, bra, identity, x, true, false;
            alg = contract_alg
        )
        x === slice || TO.tensorfree!(unnamed(x), allocator)
        TO.allocator_reset!(allocator, checkpoint)
    end
    return out
end

end
