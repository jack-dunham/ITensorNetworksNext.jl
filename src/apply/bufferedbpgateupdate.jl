using Base: @kwdef
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
