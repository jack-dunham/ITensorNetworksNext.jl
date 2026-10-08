module ITensorNetworksNextTensorOperationsExt

using Graphs: src
using ITensorBase: AbstractITensor, ITensor, inds, inputnames, mulopadd!, names, nametype,
    outputnames, rename, uniquename, unnamed
using ITensorNetworksNext: ITensorNetworksNext, BlockedMessageUpdate, BufferedBPGateUpdate,
    default_nblocks
using MatrixAlgebraKit: project_hermitian, qr_compact
using NamedGraphs: boundary_edges
using TensorAlgebra.MatrixAlgebra: sqrth_invsqrth_safe
using TensorAlgebra: TensorOperationsContract
using TensorOperations: TensorOperations as TO

# The allocator `alg` contracts with, which the intermediates must also come from.
function contract_allocator(alg::TensorOperationsContract)
    return something(alg.allocator, TO.DefaultAllocator())
end

function contract_backend(alg::TensorOperationsContract, a)
    return @something alg.backend TO.select_backend(TO.tensorcontract!, a, a, a)
end

function ITensorNetworksNext.default_nblocks(
        algorithm::BlockedMessageUpdate{<:TensorOperationsContract}, ket::AbstractArray,
        χ::Integer
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

# cuTENSOR throws `KeyError` when a contraction mixes element types, so that is rejected here
# rather than partway through a message.
function ITensorNetworksNext.check_element_types(alg::TensorOperationsContract, tensors)
    backend = contract_backend(alg, unnamed(first(tensors)))
    if backend isa TO.cuTENSORBackend && !allequal(eltype, tensors)
        throw(
            ArgumentError(
                "`BlockedMessageUpdate` on cuTENSOR needs the ket, incoming messages and " *
                    "operator to share an element type, got $(unique(map(eltype, tensors)))."
            )
        )
    end
    return nothing
end

# As in `TO.ncon`, each intermediate is an allocator temporary, freed once the next step reads it.
function ITensorNetworksNext.prod_tensors!(
        alg::TensorOperationsContract, y, x, xs...; conjlist = falses(length(xs) + 1)
    )
    allocator = contract_allocator(alg)
    checkpoint = TO.allocator_checkpoint!(allocator)
    op(i) = conjlist[i] ? conj : identity
    opx = op(1)
    for (i, m) in enumerate(Base.front(xs))
        labels = Tuple(symdiff(names(x), names(m)))
        pA, pB, pAB = TO.contract_indices(Tuple(names(x)), Tuple(names(m)), labels)
        z = TO.tensoralloc_contract(
            TO.promote_contract(eltype(x), eltype(m)), unnamed(x), pA, opx === conj,
            unnamed(m), pB, conjlist[i + 1], pAB, Val(true), allocator
        )
        z = ITensor(z, labels)
        mulopadd!(z, opx, x, op(i + 1), m, true, false; alg)
        i > 1 && TO.tensorfree!(unnamed(x), allocator)
        x, opx = z, identity
    end
    mulopadd!(y, opx, x, op(length(xs) + 1), last(xs), true, false; alg)
    length(xs) > 1 && TO.tensorfree!(unnamed(x), allocator)
    TO.allocator_reset!(allocator, checkpoint)
    return y
end

# Each chain alternates between its input and one allocator scratch with the same names;
# `mulopadd!` writes by name, so a destination needs no particular dimension order.
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
    alg = subalgorithm.contract_alg
    edges_in = [e for e in boundary_edges(env, [v]; dir = :in) if src(e) != w]
    roots = [sqrth_invsqrth_safe(project_hermitian(env[e])) for e in edges_in]
    sqrt_messages, invsqrt_messages = first.(roots), last.(roots)
    current = ψ
    if !isempty(sqrt_messages)
        allocator = contract_allocator(alg)
        checkpoint = TO.allocator_checkpoint!(allocator)
        T = mapreduce(eltype, promote_type, sqrt_messages; init = eltype(ψ))
        T === eltype(ψ) || throw(
            ArgumentError(
                "`BufferedBPGateUpdate` needs messages of eltype $(eltype(ψ)), got $T."
            )
        )
        p = (ntuple(identity, ndims(unnamed(ψ))), ())
        other = TO.tensoralloc_add(T, unnamed(ψ), p, false, Val(true), allocator)
        other = ITensor(other, names(ψ))
        for m in sqrt_messages
            bond_in = map(_ -> uniquename(nametype(ψ)), inputnames(m))
            current_in = rename(current, (inputnames(m) .=> bond_in)...)
            m_in = rename(
                m, (inputnames(m) .=> bond_in)..., (outputnames(m) .=> inputnames(m))...
            )
            mulopadd!(other, identity, current_in, identity, m_in, true, false; alg)
            current, other = other, current
        end
    end
    bondname = only(intersect(names(ψ), names(env[w => v])))
    # `qr_compact` copies an allocator-backed input, also when no permutation is needed, so
    # `Q` and `R` share no memory with the scratch (checked by pointer for both allocators).
    Q, R = qr_compact(current, setdiff(names(ψ), [bondname], names(op)))
    isempty(sqrt_messages) || TO.allocator_reset!(allocator, checkpoint)
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
    current = Q
    if !isempty(invsqrt_messages)
        allocator = contract_allocator(alg)
        checkpoint = TO.allocator_checkpoint!(allocator)
        T = mapreduce(eltype, promote_type, invsqrt_messages; init = eltype(Q))
        T === eltype(Q) || throw(
            ArgumentError(
                "`BufferedBPGateUpdate` needs messages of eltype $(eltype(Q)), got $T."
            )
        )
        p = (ntuple(identity, ndims(unnamed(Q))), ())
        other = TO.tensoralloc_add(T, unnamed(Q), p, false, Val(true), allocator)
        other = ITensor(other, names(Q))
        for m in invsqrt_messages
            bond_in = map(_ -> uniquename(nametype(Q)), inputnames(m))
            current_in = rename(current, (inputnames(m) .=> bond_in)...)
            m_in = rename(
                m, (inputnames(m) .=> bond_in)..., (outputnames(m) .=> inputnames(m))...
            )
            mulopadd!(other, identity, current_in, identity, m_in, true, false; alg)
            current, other = other, current
        end
    end
    ψ = similar(
        Q, promote_type(eltype(current), eltype(R)),
        Tuple([setdiff(inds(Q), inds(R)); setdiff(inds(R), inds(Q))])
    )
    mulopadd!(ψ, identity, current, identity, R, true, false; alg)
    isempty(invsqrt_messages) || TO.allocator_reset!(allocator, checkpoint)
    return ψ
end

end
