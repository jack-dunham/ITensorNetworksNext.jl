module ITensorNetworksNextTensorOperationsExt

using Graphs: src
using ITensorBase: ITensor, inds, name, names, state, unnamed
using ITensorNetworksNext: ITensorNetworksNext, BlockedMessageUpdate, NormNetwork, brainds,
    braname, incoming_messages, kettensor
using TensorAlgebra: TensorAlgebra as TA, TensorOperationsContract
using TensorOperations: TensorOperations as TO

promote_data(a, ::Type{T}) where {T} = eltype(a) === T ? a : T.(a)

function check_dense(a)
    a isa DenseArray && return a
    return throw(
        ArgumentError("`BlockedMessageUpdate` requires dense storage, got $(typeof(a)).")
    )
end

# Contracting a message into `labels` replaces its ket link label by the message's bra label.
function contracted_labels(labels, labels_message)
    ket_link = only(intersect(labels_message, labels))
    bra_link = only(setdiff(labels_message, (ket_link,)))
    return map(l -> l == ket_link ? bra_link : l, labels)
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
    labels_messages = map(m -> Tuple(names(m)), messages)
    labels_bra = map(n -> braname(g, n), labels_ket)
    bra_out = braname(g, ket_out)

    dim_out = findfirst(==(ket_out), labels_ket)
    χ = size(ket_data, dim_out)
    blocksize = something(algorithm.blocksize, χ)
    blocksize > 0 || throw(ArgumentError("`blocksize` must be positive, got $blocksize."))
    allocator = something(algorithm.allocator, TO.DefaultAllocator())
    alg = TensorOperationsContract(; algorithm.backend, allocator)
    # Each step swaps a ket link for its equal-length bra copy, so every intermediate has the
    # shape of the ket slice and is allocated as an unpermuted copy of it.
    same_shape = TO.trivialpermutation(ndims(ket_data), 0)

    out = similar(ket_data, T, (χ, χ))
    for first_col in 1:blocksize:χ
        cols = first_col:min(first_col + blocksize - 1, χ)
        checkpoint = TO.allocator_checkpoint!(allocator)
        x = selectdim(ket_data, dim_out, cols)
        labels = labels_ket
        for (m, labels_m) in zip(message_data, labels_messages)
            y = TO.tensoralloc_add(T, x, same_shape, false, Val(true), allocator)
            labels_next = contracted_labels(labels, labels_m)
            TA.contract!(y, labels_next, x, labels, m, labels_m; alg)
            x isa SubArray || TO.tensorfree!(x, allocator)
            x, labels = y, labels_next
        end
        TA.contractopadd!(
            view(out, :, cols), (bra_out, ket_out),
            conj, ket_data, labels_bra, identity, x, labels, true, false; alg
        )
        x isa SubArray || TO.tensorfree!(x, allocator)
        TO.allocator_reset!(allocator, checkpoint)
    end

    bra_ind = only(filter(i -> name(i) == bra_out, brainds(g)))
    ket_ind = only(filter(i -> name(i) == ket_out, inds(ket)))
    return ITensor(out, (bra_ind, ket_ind))
end

end
