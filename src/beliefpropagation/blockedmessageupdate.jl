using Graphs: src
using ITensorBase: ITensorBase, inds, inputnames, names, outputnames, state, unnamed
using TensorAlgebra: TensorOperationsContract

"""
    BlockedMessageUpdate(; normalize = true, nblocks = nothing, workspace_limit = nothing,
                           contract_alg = TensorOperationsContract())

Message update for a `NormNetwork` or a `QuadraticFormNetwork` that splits the outgoing ket leg of each message into `nblocks`
column blocks of near-equal length, so each intermediate is about `1 / nblocks` of the ket. An
`nblocks` larger than the leg's length gives one column per block, and `nblocks = nothing` chooses
it per message from the contraction backend and the ket's size. The default `contract_alg` requires
TensorOperations to be loaded. Every contraction runs with `contract_alg`, and with a
`TensorOperationsContract` intermediates are allocated and freed through its allocator.
`workspace_limit` is reserved and must be `nothing`.
"""
@kwdef struct BlockedMessageUpdate{ContractAlg} <: MessageUpdateAlgorithm
    normalize::Bool = true
    nblocks::Union{Nothing, Int} = nothing
    workspace_limit::Nothing = nothing
    contract_alg::ContractAlg = TensorOperationsContract()
    function BlockedMessageUpdate(normalize, nblocks, workspace_limit, contract_alg)
        isnothing(nblocks) || nblocks isa Integer && nblocks > 0 ||
            throw(
            ArgumentError(
                "`nblocks` must be `nothing` or a positive integer, got $nblocks."
            )
        )
        isnothing(workspace_limit) || throw(
            ArgumentError("`workspace_limit` is not supported yet and must be `nothing`.")
        )
        return new{typeof(contract_alg)}(normalize, nblocks, workspace_limit, contract_alg)
    end
end

"""
    default_nblocks(algorithm::BlockedMessageUpdate, ket::AbstractArray, χ::Integer) -> Int
    default_nblocks(backend, ketbytes::Integer, χ::Integer) -> Int

The number of column blocks `algorithm` splits a leg of length `χ` of the ket array `ket` into;
the kernel caps it at `χ`. The first form returns `algorithm.nblocks` when it is set; for `nothing`
and a `TensorOperationsContract` it finds the TensorOperations backend `algorithm.contract_alg`
contracts with and calls the second, which a backend overloads, and otherwise returns 1. The
second form is defined by the TensorOperations extension: 1 (the whole leg) by default, and on
cuTENSOR about 16 blocks, each at least 4 MiB and at most 64 columns.
"""
function default_nblocks end
function default_nblocks(algorithm::BlockedMessageUpdate, ket::AbstractArray, χ::Integer)
    return @something algorithm.nblocks 1
end

# Rejects operands whose element types the backend `alg` contracts with cannot mix, before any
# contraction runs. The TensorOperations extension overloads it for cuTENSOR.
check_element_types(alg, tensors) = nothing

function message_update!(
        algorithm::BlockedMessageUpdate, cache, factors::AbstractBilinearFormNetwork, edge
    )
    return bilinearform_message_update!(algorithm, cache, factors, edge)
end

# The ket and the tensors each block contracts into its slice in turn, checked before any
# contraction runs.
function message_contraction_tensors(algorithm, factors::NormNetwork, messages, vertex)
    rest = map(state, collect(messages))
    return checked_tensors(algorithm, kettensor(factors, vertex), rest)
end
# The operator is one more step of the block contraction, so it may carry only its paired site
# indices: a link index to a neighbouring operator would be a third leg on the message.
function message_contraction_tensors(
        algorithm, factors::QuadraticFormNetwork, messages, vertex
    )
    op = operatornetwork(factors)[vertex]

    linkinds = setdiff(names(op), [inputnames(op); outputnames(op)])

    if !isempty(linkinds)
        throw(
            ArgumentError(
                "`BlockedMessageUpdate` needs a product operator, but the operator has " *
                    "indices $linkinds besides its inputs and outputs."
            )
        )
    end

    rest = [operatortensor(factors, vertex); map(state, collect(messages))]

    return checked_tensors(algorithm, kettensor(factors, vertex), rest)
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

function updated_message(
        algorithm::BlockedMessageUpdate, cache, factors::AbstractBilinearFormNetwork, edge
    )
    vertex = src(edge)
    messages = incoming_messages(cache, edge)

    ket, rest = message_contraction_tensors(algorithm, factors, messages, vertex)

    # Shares the ket's data; the closing contraction conjugates it through its `conj` op.
    bra = conj_bratensor(factors, vertex)
    # The far vertex of `edge` may not be in `factors`, so the leg is found on the message.
    ketdimname = only(intersect(names(cache[edge]), names(ket)))

    χ = size(ket, ITensorBase.findname(ket, ketdimname))

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

