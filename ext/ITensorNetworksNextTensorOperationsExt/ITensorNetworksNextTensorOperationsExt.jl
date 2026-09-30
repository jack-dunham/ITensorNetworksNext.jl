module ITensorNetworksNextTensorOperationsExt

using Graphs: src
using ITensorBase: ITensorBase, ITensor, names, rename, state, unnamed
using ITensorNetworksNext: ITensorNetworksNext, BlockedMessageUpdate, NormNetwork, braname,
    default_nblocks, incoming_messages, kettensor
using TensorAlgebra: TensorAlgebra as TA, TensorOperationsContract
using TensorOperations: TensorOperations as TO

function check_dense(t)
    a = unnamed(t)
    a isa DenseArray || throw(
        ArgumentError("`BlockedMessageUpdate` requires dense storage, got $(typeof(a)).")
    )
    return t
end

# cuTENSOR throws `KeyError` when a contraction mixes element types, so reject that up front.
check_eltypes(backend, tensors) = nothing
function check_eltypes(::TO.cuTENSORBackend, tensors)
    allequal(eltype, tensors) || throw(
        ArgumentError(
            "`BlockedMessageUpdate` on cuTENSOR needs the ket and incoming messages to share an " *
                "element type, got $(unique(map(eltype, tensors)))."
        )
    )
    return nothing
end

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

# Contracting a message into `x` replaces the ket link name they share by the message's bra name.
function contracted_names(x, message)
    ket_link = only(intersect(names(message), names(x)))
    bra_link = only(setdiff(names(message), (ket_link,)))
    return map(n -> n == ket_link ? bra_link : n, names(x))
end

function ITensorNetworksNext.updated_message(
        algorithm::BlockedMessageUpdate, cache, factors::NormNetwork, edge
    )
    g = factors[src(edge)]
    ket = check_dense(kettensor(g))
    messages = map(m -> check_dense(state(m)), collect(incoming_messages(cache, edge)))
    alg = algorithm.contract_alg
    check_eltypes(contract_backend(alg, unnamed(ket)), [ket; messages])
    T = promote_type(eltype(ket), map(eltype, messages)...)
    # Shares the ket's data; the closing contraction conjugates it through its `conj` op.
    bra = rename(n -> braname(g, n), ket)
    # The far vertex of `edge` may not be in `factors`, so the leg is found on the message.
    ket_out = only(intersect(names(cache[edge]), names(ket)))

    χ = size(ket, ITensorBase.dim(ket, ket_out))
    nblocks = min(default_nblocks(algorithm, unnamed(ket), χ), χ)
    allocator = contract_allocator(alg)
    # Each step swaps a ket link for its equal-length bra copy, so every intermediate has the
    # shape of the ket slice and is allocated as an unpermuted copy of it.
    same_shape = TO.trivialpermutation(ndims(ket), 0)

    out = ITensor(similar(unnamed(ket), T, (χ, χ)), (braname(g, ket_out), ket_out))
    for block in 1:nblocks
        cols = (fld((block - 1) * χ, nblocks) + 1):fld(block * χ, nblocks)
        checkpoint = TO.allocator_checkpoint!(allocator)
        slice = view(ket, ket_out => cols)
        x = slice
        for m in messages
            y = ITensor(
                TO.tensoralloc_add(T, unnamed(x), same_shape, false, Val(true), allocator),
                contracted_names(x, m)
            )
            TA.contract!(y, x, m; alg)
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
