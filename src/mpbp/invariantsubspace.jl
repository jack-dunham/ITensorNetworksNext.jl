using AlgorithmsInterface: AlgorithmsInterface as AI
using ITensorBase: AbstractNamedTensor, Index, commoninds, inds, name, names, rename
using LinearAlgebra: I, cond, diag, ordschur, schur
using MatrixAlgebraKit: MatrixAlgebraKit, eig_trunc
using TensorAlgebra: matricize, unmatricize

@kwdef struct DenseEig <: AbstractAlgorithm
    rtol::Float64 = 1.0e-12
    degeneracy_rtol::Float64 = 1.0e-10
end

function invariant_subspace end

struct InvariantSubspaceProblem{Operator} <: AI.Problem
    matrix::Operator
    maxdim::Int
end

function default_algorithm(::typeof(invariant_subspace), ::Type{<:Tuple}; kwargs...)
    return DenseEig(; kwargs...)
end

"""
    CornerTransferProduct(tensors)

The product of named `tensors`, contracted over the cuts they share. It maps the cut where the
last tensor meets the first to itself, in the order of the last one's legs.
"""
struct CornerTransferProduct{F <: AbstractNamedTensor, C <: Tuple}
    factors::Vector{F}
    cut::C
end
function CornerTransferProduct(tensors::Vector{<:AbstractNamedTensor})
    return CornerTransferProduct(tensors, Tuple(commoninds(last(tensors), first(tensors))))
end

function Base.eltype(product::CornerTransferProduct)
    return mapreduce(eltype, promote_type, product.factors)
end

function Base.Matrix(product::CornerTransferProduct)
    # The first factor's copy of `cut` is renamed, so the last factor does not contract with it.
    output = map(index -> Index(length(index)), product.cut)
    first_factor = rename(first(product.factors), (product.cut .=> output)...)
    closed = foldl(*, product.factors[2:end]; init = first_factor)
    return matricize(closed, output, product.cut)
end

# Keeps the eigenvalues of largest modulus, sorted by it: at most `maxdim`, above `rtol` relative
# to the largest, and never splitting eigenvalues of equal modulus.
struct KeepDominant <: MatrixAlgebraKit.TruncationStrategy
    maxdim::Int
    rtol::Float64
    degeneracy_rtol::Float64
end
KeepDominant(alg, maxdim) = KeepDominant(maxdim, alg.rtol, alg.degeneracy_rtol)

function MatrixAlgebraKit.findtruncated(values::AbstractVector, strategy::KeepDominant)
    order = sortperm(abs.(values); rev = true)
    moduli = abs.(values[order])
    nkept = min(strategy.maxdim, count(>(strategy.rtol * first(moduli)), moduli))
    while 0 < nkept < length(moduli) &&
            moduli[nkept] - moduli[nkept + 1] ≤ strategy.degeneracy_rtol * moduli[nkept]
        nkept -= 1
    end
    nkept > 0 || throw(
        ArgumentError(
            "`maxdim = $(strategy.maxdim)` splits the dominant eigenvalue multiplet; " *
                "raise `maxdim`."
        )
    )
    return order[1:nkept]
end

# Rows spanning the left invariant subspace of `matrix` for its `nkept` eigenvalues of largest
# modulus, from an ordered Schur form, so that no eigenvectors of the discarded ones enter.
function left_invariant_rows(matrix, nkept)
    decomposition = schur(Matrix(transpose(matrix)))
    moduli = sort(abs.(decomposition.values); rev = true)
    threshold = if nkept < length(moduli)
        (moduli[nkept] + moduli[nkept + 1]) / 2
    else
        -one(eltype(moduli))
    end
    decomposition = ordschur(decomposition, abs.(decomposition.values) .> threshold)
    return transpose(decomposition.Z[:, 1:nkept])
end

# `left * right` as a matrix. Named bases keep their kept index first in `left` and last in
# `right`, and the matrix runs over those two.
overlap_matrix(left::AbstractMatrix, right::AbstractMatrix) = left * right
function overlap_matrix(left::AbstractNamedTensor, right::AbstractNamedTensor)
    return matricize(left * right, (first(inds(left)),), (last(inds(right)),))
end

# `(right * transform, transform \ left)`, with `transform` acting on the bases' kept index.
function transformed(right::AbstractMatrix, left::AbstractMatrix, transform)
    return (right * transform, transform \ left)
end
function transformed(right::AbstractNamedTensor, left::AbstractNamedTensor, transform)
    old, new = last(inds(right)), Index(size(transform, 2))
    return right * unmatricize(transform, (old,), (new,)),
        unmatricize(inv(transform), (new,), (old,)) * left
end

# Matches each new eigenvector to the previous basis vector it overlaps most, then rotates the
# bases onto `previous` within blocks of equal eigenvalues, which leaves the corners diagonal.
function align_bases(right_basis, left_basis, eigenvalues, previous, degeneracy_rtol)
    size(previous) == size(right_basis) || return right_basis, left_basis, eigenvalues
    overlaps = overlap_matrix(left_basis, previous)
    order = Int[]
    for column in axes(overlaps, 2)
        candidates = setdiff(axes(overlaps, 1), order)
        push!(order, candidates[argmax(abs.(overlaps[candidates, column]))])
    end
    eigenvalues = eigenvalues[order]
    permutation = Matrix{Float64}(I, length(order), length(order))[:, order]

    # The overlap of the permuted bases with `previous`.
    rotation = overlaps[order, :]
    for i in axes(rotation, 1), j in axes(rotation, 2)
        degenerate =
            i == j ||
            abs(eigenvalues[i] - eigenvalues[j]) ≤ degeneracy_rtol * abs(eigenvalues[i])
        degenerate || (rotation[i, j] = 0)
    end
    # A near-singular rotation means the subspace has moved; aligning onto it would amplify noise.
    transform = cond(rotation) < 1.0e8 ? permutation * rotation : permutation
    return transformed(right_basis, left_basis, transform)..., eigenvalues
end

"""
    invariant_subspace(alg, matrix, maxdim; reference = nothing)
        -> (right_basis, left_basis, eigenvalues)

Dominant invariant subspace of `matrix`: `matrix * right_basis ≈ right_basis * Diagonal(eigenvalues)`,
`left_basis * matrix ≈ Diagonal(eigenvalues) * left_basis` and `left_basis * right_basis ≈ I`.
Eigenvalues of equal modulus are kept or dropped together.

With `reference = (right_basis, left_basis)` from a previous call of the same size, the right
basis is rotated onto the reference's within blocks of eigenvalues equal to relative tolerance
`alg.degeneracy_rtol`, and the eigenvalues follow the reference's order.

For a `CornerTransferProduct` with cut `cut`, `right_basis` has legs `(cut..., k)` and
`left_basis` legs `(k, cut...)` for a new index `k`. A `reference` of that form is used only
when its cut has the names of `cut`.
"""
function invariant_subspace(alg::DenseEig, matrix, maxdim::Integer; reference = nothing)
    result = AI.solve(InvariantSubspaceProblem(matrix, maxdim), DenseEigensolve(alg))
    return aligned(result, reference, alg)
end
function invariant_subspace(
        alg::DenseEig, product::CornerTransferProduct, maxdim::Integer; reference = nothing
    )
    cut = product.cut
    right_basis, left_basis, eigenvalues = invariant_subspace(alg, Matrix(product), maxdim)
    k = Index(length(eigenvalues))
    result = (
        unmatricize(right_basis, cut, (k,)),
        unmatricize(left_basis, (k,), cut),
        eigenvalues,
    )
    return aligned(result, usable_reference(reference, cut), alg)
end

# `reference` when its legs other than the kept index have the names of `cut`, else `nothing`.
function usable_reference(reference, cut)
    isnothing(reference) && return nothing
    right_basis = first(reference)
    issetequal(names(right_basis), (name.(cut)..., name(last(inds(right_basis))))) ||
        return nothing
    return reference
end

function aligned(result, reference, alg)
    isnothing(reference) && return result
    return align_bases(result..., first(reference), alg.degeneracy_rtol)
end

# One dense eigendecomposition, run as a single step; the iterate is the kept
# `(right_basis, left_basis, eigenvalues)` once the step has run.
struct DenseEigensolve <: AI.Algorithm
    dense_eig::DenseEig
    stopping_criterion::AI.StopAfterIteration
end
DenseEigensolve(dense_eig::DenseEig) = DenseEigensolve(dense_eig, AI.StopAfterIteration(1))

@kwdef mutable struct DenseEigensolveState{StoppingCriterionState} <: AI.State
    iterate = nothing
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

function AI.initialize_state(
        problem::InvariantSubspaceProblem, algorithm::DenseEigensolve; kwargs...
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion
    )
    return DenseEigensolveState(; stopping_criterion_state)
end

function AI.step!(
        problem::InvariantSubspaceProblem, algorithm::DenseEigensolve,
        state::DenseEigensolveState
    )
    (; matrix, maxdim) = problem
    dense = Matrix(matrix)
    values, right_basis =
        eig_trunc(dense; trunc = KeepDominant(algorithm.dense_eig, maxdim))
    left_rows = left_invariant_rows(dense, size(right_basis, 2))
    state.iterate = (right_basis, (left_rows * right_basis) \ left_rows, diag(values))
    return state
end
