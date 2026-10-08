using Base: @kwdef
using TensorAlgebra: TensorOperationsContract

"""
    BufferedBPGateUpdate(; contract_alg = TensorOperationsContract())

The gate stages of [`BPApplyGate`](@ref) that take their scratch tensors from the allocator
of `contract_alg` instead of allocating one per step, for dense storage only; other storage
throws an `ArgumentError`. Its stages are defined only when TensorOperations is loaded.
"""
@kwdef struct BufferedBPGateUpdate{ContractAlg}
    contract_alg::ContractAlg = TensorOperationsContract()
end
