using Base: @kwdef
using Graphs: src
using ITensorBase: AbstractITensor, inds, inputnames, mulopadd!, names, nametype, operator,
    rename, uniquename, unnamed
using MatrixAlgebraKit: qr_compact
using NamedGraphs: boundary_edges
using TensorAlgebra: TensorOperationsContract

"""
    BufferedBPGateUpdate(; contract_alg = TensorOperationsContract())

The gate stages of [`BPApplyGate`](@ref) that apply each message chain in place with
[`absorb_matrices!`](@ref), holding two vertex-sized tensors, for dense storage only; other
storage throws an `ArgumentError`. Every contraction uses `contract_alg`; the default needs
TensorOperations loaded.
"""
@kwdef struct BufferedBPGateUpdate{ContractAlg}
    contract_alg::ContractAlg = TensorOperationsContract()
end

# The `message_gauge` pair as operators from the bond name back to the bond name, so
# `absorb_matrices!` leaves `ψ`'s names unchanged while the leg holds the rank space.
function gauge_operators(message)
    x, y = message_gauge(message)
    bond = only(inputnames(message))
    rank = only(setdiff(names(x), [bond]))
    out = uniquename(nametype(y))
    root = operator(x, (rank,), (bond,))
    inverse_root = operator(rename(y, bond => out, rank => bond), (out,), (bond,))
    return root, inverse_root
end

function bp_gate_factorize!(
        subalgorithm::BufferedBPGateUpdate{<:TensorOperationsContract}, op::AbstractITensor,
        state, env, v, w
    )
    ψ = state[v]
    unnamed(ψ) isa DenseArray || throw(
        ArgumentError(
            "`BufferedBPGateUpdate` requires dense storage, got $(typeof(unnamed(ψ)))."
        )
    )
    edges_in = [e for e in boundary_edges(env, [v]; dir = :in) if src(e) != w]
    gauges = [gauge_operators(env[e]) for e in edges_in]
    absorb_matrices!(subalgorithm.contract_alg, ψ, first.(gauges)...)
    bondname = only(intersect(names(ψ), names(env[w => v])))
    Q, R = qr_compact(ψ, setdiff(names(ψ), [bondname], names(op)))
    return Q, R, last.(gauges)
end

function bp_gate_restore!(
        subalgorithm::BufferedBPGateUpdate{<:TensorOperationsContract},
        Q::AbstractITensor, R::AbstractITensor, inverse_roots
    )
    unnamed(Q) isa DenseArray || throw(
        ArgumentError(
            "`BufferedBPGateUpdate` requires dense storage, got $(typeof(unnamed(Q)))."
        )
    )
    alg = subalgorithm.contract_alg
    absorb_matrices!(alg, Q, inverse_roots...)
    ψ = similar(
        Q, promote_type(eltype(Q), eltype(R)),
        Tuple([setdiff(inds(Q), inds(R)); setdiff(inds(R), inds(Q))])
    )
    return mulopadd!(ψ, identity, Q, identity, R, true, false; alg)
end
