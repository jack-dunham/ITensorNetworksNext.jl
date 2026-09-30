module ITensorNetworksNextTensorOperationsExt

using Graphs: src
using ITensorBase: ITensorBase, inds, names, rename, state, unnamed
using ITensorNetworksNext: ITensorNetworksNext, BlockedMessageUpdate, NormNetwork, braname,
    check_input, default_nblocks, incoming_messages, kettensor, updated_message
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

# Non-dense storage cannot be sliced by column. cuTENSOR throws `KeyError` when a contraction
# mixes element types, so that is rejected here rather than partway through a message.
function ITensorNetworksNext.check_input(
        ::typeof(updated_message), algorithm::BlockedMessageUpdate, cache,
        factors::NormNetwork, edge
    )
    ket = kettensor(factors[src(edge)])
    tensors = [ket; map(state, collect(incoming_messages(cache, edge)))]
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
                "`BlockedMessageUpdate` on cuTENSOR needs the ket and incoming messages to " *
                    "share an element type, got $(unique(map(eltype, tensors)))."
            )
        )
    end
    return nothing
end

function ITensorNetworksNext.updated_message(
        algorithm::BlockedMessageUpdate, cache, factors::NormNetwork, edge
    )
    check_input(updated_message, algorithm, cache, factors, edge)
    g = factors[src(edge)]
    ket = kettensor(g)
    messages = map(state, collect(incoming_messages(cache, edge)))
    alg = algorithm.contract_alg
    T = promote_type(eltype(ket), map(eltype, messages)...)
    # Shares the ket's data; the closing contraction conjugates it through its `conj` op.
    bra = rename(n -> braname(g, n), ket)
    # The far vertex of `edge` may not be in `factors`, so the leg is found on the message.
    ket_out = only(intersect(names(cache[edge]), names(ket)))

    χ = size(ket, ITensorBase.dim(ket, ket_out))
    nblocks = min(default_nblocks(algorithm, unnamed(ket), χ), χ)
    allocator = contract_allocator(alg)
    step_alg = temporary_alg(alg)

    # The message being replaced has the output's indices, bra copy then ket leg.
    out = similar(ket, T, Tuple(inds(state(cache[edge]))))
    for block in 1:nblocks
        cols = (fld((block - 1) * χ, nblocks) + 1):fld(block * χ, nblocks)
        checkpoint = TO.allocator_checkpoint!(allocator)
        slice = view(ket, ket_out => cols)
        x = slice
        for m in messages
            y = TA.contract(x, m; alg = step_alg)
            x === slice || TO.tensorfree!(unnamed(x), allocator)
            x = y
        end
        TA.contractopadd!(
            view(out, ket_out => cols), conj, bra, identity, x, true, false; alg
        )
        x === slice || TO.tensorfree!(unnamed(x), allocator)
        TO.allocator_reset!(allocator, checkpoint)
    end
    return out
end

end
