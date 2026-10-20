module ITensorNetworksNextTensorOperationsExt

using ITensorBase: AbstractITensor, ITensor, mulopadd!, names, unnamed
using ITensorNetworksNext: ITensorNetworksNext
using TensorAlgebra: TensorOperationsContract, contractadd!
using TensorOperations: TensorOperations as TO

# The allocator `alg` contracts with, which the intermediates must also come from.
function contract_allocator(alg::TensorOperationsContract)
    return something(alg.allocator, TO.DefaultAllocator())
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

# The labels after each absorption are worked out here, so that a method for the data's
# array type sees only arrays and labels.
function ITensorNetworksNext.absorb_matrices!(
        alg::TensorOperationsContract, t::AbstractITensor, matrices::AbstractITensor...
    )
    isempty(matrices) && return t

    labels_dests = accumulate(matrices; init = names(t)) do labels, m
        shared, added = intersect(names(m), labels), setdiff(names(m), labels)
        return replace(labels, only(shared) => only(added))
    end

    a = ITensorNetworksNext.absorb_matrices!(
        alg, unnamed(t), names(t), unnamed.(matrices), names.(matrices), labels_dests
    )
    return ITensor(a, last(labels_dests))
end

# TensorOperations contracts only dense arrays; graded arrays, for example, are read element
# by element.
function ITensorNetworksNext.absorb_matrices!(
        alg::TensorOperationsContract, a::AbstractArray, labels_a,
        matrices::Tuple, labels_matrices::Tuple, labels_dests::Tuple
    )
    throw(
        ArgumentError(
            "`absorb_matrices!` with `TensorOperationsContract` needs dense arrays, got " *
                "$(typeof(a))."
        )
    )
end

function ITensorNetworksNext.check_input(
        ::typeof(ITensorNetworksNext.absorb_matrices!), alg::TensorOperationsContract,
        a::DenseArray, matrices::Tuple
    )
    for matrix in matrices
        if !(matrix isa AbstractMatrix)
            throw(
                ArgumentError(
                    "`absorb_matrices!` with `TensorOperationsContract` needs matrices, " *
                        "got an array of dimension $(ndims(matrix))."
                )
            )
        elseif size(matrix, 1) != size(matrix, 2)
            throw(
                ArgumentError(
                    "`absorb_matrices!` with `TensorOperationsContract` needs square " *
                        "matrices, got size $(size(matrix))."
                )
            )
        elseif eltype(a) !== eltype(matrix)
            throw(
                ArgumentError(
                    "`absorb_matrices!` with `TensorOperationsContract` needs matrices of " *
                        "the array's element type $(eltype(a)), got $(eltype(matrix))."
                )
            )
        end
    end
    return nothing
end

# The product is formed in `a`'s memory and one allocator scratch array, freed on return,
# which needs square matrices of `a`'s element type.
function ITensorNetworksNext.absorb_matrices!(
        alg::TensorOperationsContract, a::DenseArray, labels_a,
        matrices::Tuple, labels_matrices::Tuple, labels_dests::Tuple
    )
    ITensorNetworksNext.check_input(ITensorNetworksNext.absorb_matrices!, alg, a, matrices)

    allocator = contract_allocator(alg)
    checkpoint = TO.allocator_checkpoint!(allocator)
    p = (ntuple(identity, ndims(a)), ())
    scratch = TO.tensoralloc_add(eltype(a), a, p, false, Val(true), allocator)

    x, other, labels_x = a, scratch, labels_a
    for (m, labels_m, labels_y) in zip(matrices, labels_matrices, labels_dests)
        y = other

        # cuTENSOR 2.7 rejects outputs of 2^30 elements whose last extent has no small
        # factor, so those are written in blocks of the last leg, which `x` or `m` carries.
        if length(y) < 2^30
            contractadd!(y, labels_y, x, labels_x, m, labels_m, true, false; alg)
        else
            leg, ncols = last(labels_y), size(y, ndims(y))
            dim_x, dim_m = findfirst(==(leg), labels_x), findfirst(==(leg), labels_m)
            step = max(1, (2^30 - 1) ÷ (length(y) ÷ ncols))

            for cols in Iterators.partition(1:ncols, step)
                xb = isnothing(dim_x) ? x : selectdim(x, dim_x, cols)
                mb = isnothing(dim_m) ? m : selectdim(m, dim_m, cols)
                yb = selectdim(y, ndims(y), cols)
                contractadd!(yb, labels_y, xb, labels_x, mb, labels_m, true, false; alg)
            end
        end

        x, other, labels_x = y, x, labels_y
    end
    isodd(length(matrices)) && copyto!(a, x)

    TO.tensorfree!(scratch, allocator)
    TO.allocator_reset!(allocator, checkpoint)
    return a
end

end
