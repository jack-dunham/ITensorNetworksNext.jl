using Base: @kwdef
using Graphs: src
using ITensorBase: AbstractITensor, inds, mulopadd!, names, unnamed
using MatrixAlgebraKit: project_hermitian, qr_compact
using NamedGraphs: boundary_edges
using TensorAlgebra.MatrixAlgebra: sqrth_invsqrth_safe
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

function ITensorNetworksNext.bp_gate_factorize!(
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
    roots = [sqrth_invsqrth_safe(project_hermitian(env[e])) for e in edges_in]
    sqrt_messages, invsqrt_messages = first.(roots), last.(roots)
    absorb_matrices!(subalgorithm.contract_alg, ψ, sqrt_messages...)
    bondname = only(intersect(names(ψ), names(env[w => v])))
    Q, R = qr_compact(ψ, setdiff(names(ψ), [bondname], names(op)))
    return Q, R, invsqrt_messages
end

function ITensorNetworksNext.bp_gate_restore!(
        subalgorithm::BufferedBPGateUpdate{<:TensorOperationsContract},
        Q::AbstractITensor, R::AbstractITensor, invsqrt_messages
    )
    unnamed(Q) isa DenseArray || throw(
        ArgumentError(
            "`BufferedBPGateUpdate` requires dense storage, got $(typeof(unnamed(Q)))."
        )
    )
    alg = subalgorithm.contract_alg
    absorb_matrices!(alg, Q, invsqrt_messages...)
    ψ = similar(
        Q, promote_type(eltype(Q), eltype(R)),
        Tuple([setdiff(inds(Q), inds(R)); setdiff(inds(R), inds(Q))])
    )
    return mulopadd!(ψ, identity, Q, identity, R, true, false; alg)
end
