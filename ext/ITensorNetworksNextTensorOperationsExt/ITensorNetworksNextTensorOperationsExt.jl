module ITensorNetworksNextTensorOperationsExt

using Graphs: src
using ITensorBase: ITensor, inds, name, names, state, unnamed
using ITensorNetworksNext: ITensorNetworksNext, BlockedMessageUpdate, NormNetwork, brainds,
    braname, incoming_messages, kettensor
using TensorOperations: TensorOperations as TO

promote_data(a, ::Type{T}) where {T} = eltype(a) === T ? a : T.(a)

function check_dense(a)
    a isa DenseArray && return a
    return throw(
        ArgumentError("`BlockedMessageUpdate` requires dense storage, got $(typeof(a)).")
    )
end

# Index tuples for contracting each incoming message into the ket in turn, each step replacing a
# ket link label by its bra copy, and for the closing contraction against the conjugated ket.
function contraction_steps(g, labels_ket, labels_messages, ket_out)
    steps = []
    labels = labels_ket
    for labels_message in labels_messages
        ket_link = only(intersect(labels_message, labels))
        bra_link = only(setdiff(labels_message, (ket_link,)))
        labels_next = map(l -> l == ket_link ? bra_link : l, labels)
        push!(steps, TO.contract_indices(labels, labels_message, labels_next))
        labels = labels_next
    end
    labels_bra = map(n -> braname(g, n), labels_ket)
    closing = TO.contract_indices(labels_bra, labels, (braname(g, ket_out), ket_out))
    return steps, closing
end

function ITensorNetworksNext.updated_message(
        algorithm::BlockedMessageUpdate, cache, factors::NormNetwork, edge
    )
    isnothing(algorithm.workspace_limit) || throw(
        ArgumentError("`workspace_limit` is not supported yet and must be `nothing`.")
    )
    g = factors[src(edge)]
    ket = kettensor(g)
    labels_ket = Tuple(names(ket))
    # The far vertex of `edge` may not be in `factors`, so the leg is found on the message.
    ket_out = only(intersect(names(cache[edge]), labels_ket))
    messages = map(state, collect(incoming_messages(cache, edge)))
    T = promote_type(eltype(ket), map(eltype, messages)...)
    ket_data = promote_data(check_dense(unnamed(ket)), T)
    message_data = map(m -> promote_data(check_dense(unnamed(m)), T), messages)
    steps, closing = contraction_steps(
        g, labels_ket, map(m -> Tuple(names(m)), messages), ket_out
    )

    dim_out = findfirst(==(ket_out), labels_ket)
    χ = size(ket_data, dim_out)
    blocksize = something(algorithm.blocksize, χ)
    blocksize > 0 || throw(ArgumentError("`blocksize` must be positive, got $blocksize."))
    backend = something(algorithm.backend, TO.DefaultBackend())
    allocator = something(algorithm.allocator, TO.DefaultAllocator())

    out = similar(ket_data, T, (χ, χ))
    for first_col in 1:blocksize:χ
        cols = first_col:min(first_col + blocksize - 1, χ)
        checkpoint = TO.allocator_checkpoint!(allocator)
        x = selectdim(ket_data, dim_out, cols)
        for (m, (pA, pB, pAB)) in zip(message_data, steps)
            y = TO.tensoralloc_contract(
                T, x, pA, false, m, pB, false, pAB, Val(true), allocator
            )
            TO.tensorcontract!(
                y, x, pA, false, m, pB, false, pAB, TO.One(), TO.Zero(), backend, allocator
            )
            x isa SubArray || TO.tensorfree!(x, allocator)
            x = y
        end
        pA, pB, pAB = closing
        TO.tensorcontract!(
            view(out, :, cols), ket_data, pA, true, x, pB, false, pAB,
            TO.One(), TO.Zero(), backend, allocator
        )
        x isa SubArray || TO.tensorfree!(x, allocator)
        TO.allocator_reset!(allocator, checkpoint)
    end

    bra_ind = only(filter(i -> name(i) == braname(g, ket_out), brainds(g)))
    ket_ind = only(filter(i -> name(i) == ket_out, inds(ket)))
    return ITensor(out, (bra_ind, ket_ind))
end

end
