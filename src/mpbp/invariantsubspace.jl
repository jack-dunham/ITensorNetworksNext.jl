using AlgorithmsInterface: AlgorithmsInterface as AI
using ITensorBase: AbstractNamedTensor, Index, commoninds, inds, name, names, rename
using KrylovKit: Arnoldi, eigsolve, schursolve
using LinearAlgebra: I, cond, diag, norm, ordschur, schur
using MatrixAlgebraKit: MatrixAlgebraKit, eig_trunc
using Random: Xoshiro
using TensorAlgebra: matricize, unmatricize

@kwdef struct DenseEig <: AbstractAlgorithm
    rtol::Float64 = 1.0e-12
    degeneracy_rtol::Float64 = 1.0e-10
end

"""
    KrylovEig(; rtol, degeneracy_rtol, tol, krylovdim, maxiter)

Finds the dominant invariant subspace of a `CornerTransferProduct` by Arnoldi iteration, applying
the product's factors one at a time instead of forming it. `rtol` and `degeneracy_rtol` truncate
as for `DenseEig`; `tol`, relative to the size of the product, `krylovdim` and `maxiter` are
passed to KrylovKit's `Arnoldi`. Arnoldi iteration from one starting vector can return a repeated
eigenvalue fewer times than it occurs, which `DenseEig` does not.
"""
@kwdef struct KrylovEig <: AbstractAlgorithm
    rtol::Float64 = 1.0e-12
    degeneracy_rtol::Float64 = 1.0e-10
    tol::Float64 = 1.0e-12
    krylovdim::Int = 30
    maxiter::Int = 100
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

# `product` applied to the vector `x` over its cut, or `x` applied to `product` when `transposed`.
function apply_product(
        product::CornerTransferProduct,
        x::AbstractVector;
        transposed = false
    )
    column = Index(1)
    tensor = unmatricize(reshape(x, :, 1), product.cut, (column,))
    result = if transposed
        foldl(*, product.factors; init = tensor)
    else
        foldr(*, product.factors; init = tensor)
    end
    return vec(matricize(result, product.cut, (column,)))
end

# Keeps the eigenvalues of largest modulus, sorted by it: at most `maxdim`, above `rtol` relative
# to the largest, and never splitting moduli equal to `degeneracy_rtol` plus `degeneracy_atol`.
struct KeepDominant <: MatrixAlgebraKit.TruncationStrategy
    maxdim::Int
    rtol::Float64
    degeneracy_rtol::Float64
    degeneracy_atol::Float64
end
function KeepDominant(alg, maxdim; degeneracy_atol = 0.0)
    return KeepDominant(maxdim, alg.rtol, alg.degeneracy_rtol, degeneracy_atol)
end

function MatrixAlgebraKit.findtruncated(values::AbstractVector, strategy::KeepDominant)
    order = sortperm(abs.(values); rev = true)
    moduli = abs.(values[order])
    nkept = min(strategy.maxdim, count(>(strategy.rtol * first(moduli)), moduli))
    while 0 < nkept < length(moduli) &&
            moduli[nkept] - moduli[nkept + 1] ≤
            strategy.degeneracy_rtol * moduli[nkept] + strategy.degeneracy_atol
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
    result = AI.solve(InvariantSubspaceProblem(matrix, maxdim), Eigensolve(alg))
    return aligned(result, reference, alg)
end
function invariant_subspace(
        alg::DenseEig, product::CornerTransferProduct, maxdim::Integer; reference = nothing
    )
    result = invariant_subspace(alg, Matrix(product), maxdim)
    return named_subspace(result, product.cut, alg, reference)
end
function invariant_subspace(
        alg::KrylovEig, product::CornerTransferProduct, maxdim::Integer; reference = nothing
    )
    result = AI.solve(InvariantSubspaceProblem(product, maxdim), Eigensolve(alg))
    return named_subspace(result, product.cut, alg, reference)
end

# The matrix bases of `result` as named tensors over `cut`, aligned onto `reference`.
function named_subspace((right_basis, left_basis, eigenvalues), cut, alg, reference)
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

# One eigensolve by `alg`, run as a single step; the iterate is the kept
# `(right_basis, left_basis, eigenvalues)` once the step has run.
struct Eigensolve{Alg <: AbstractAlgorithm} <: AI.Algorithm
    alg::Alg
    stopping_criterion::AI.StopAfterIteration
end
Eigensolve(alg::AbstractAlgorithm) = Eigensolve(alg, AI.StopAfterIteration(1))

@kwdef mutable struct EigensolveState{StoppingCriterionState} <: AI.State
    iterate = nothing
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

function AI.initialize_state(
        problem::InvariantSubspaceProblem,
        algorithm::Eigensolve;
        kwargs...
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion
    )
    return EigensolveState(; stopping_criterion_state)
end

function AI.step!(
        problem::InvariantSubspaceProblem, algorithm::Eigensolve{DenseEig},
        state::EigensolveState
    )
    (; matrix, maxdim) = problem
    state.iterate = dense_subspace(Matrix(matrix), KeepDominant(algorithm.alg, maxdim))
    return state
end

function AI.step!(
        problem::InvariantSubspaceProblem, algorithm::Eigensolve{KrylovEig},
        state::EigensolveState
    )
    (; matrix, maxdim) = problem
    alg = algorithm.alg
    n = prod(length, matrix.cut)
    # One more eigenvalue than kept shows whether `maxdim` splits a multiplet.
    howmany = maxdim + 1
    krylovdim = max(alg.krylovdim, 2 * howmany)
    # A space no larger than the Krylov space is solved densely.
    if n ≤ krylovdim
        state.iterate = dense_subspace(Matrix(matrix), KeepDominant(alg, maxdim))
        return state
    end
    # A fixed seed gives every rank, and a serial run, the same starting vector.
    x0 = randn(Xoshiro(0), eltype(matrix), n)
    # KrylovKit's `tol` bounds the residual norm, so it is scaled to the size of the product.
    scale = norm(apply_product(matrix, x0)) / norm(x0)
    arnoldi =
        Arnoldi(; tol = alg.tol * scale, krylovdim, maxiter = alg.maxiter, verbosity = 0)

    values, vectors, info =
        eigsolve(x -> apply_product(matrix, x), x0, howmany, :LM, arnoldi)
    check_converged(info, howmany, alg.maxiter)
    # Eigenvalues are accurate to about the residual bound, so moduli that close count as equal.
    trunc = KeepDominant(alg, maxdim; degeneracy_atol = alg.tol * scale)
    kept = MatrixAlgebraKit.findtruncated(values[1:info.converged], trunc)
    right_basis = stack(vectors[kept])

    # Schur vectors of the transpose span the left invariant subspace without eigenvectors
    # of the discarded eigenvalues, as `left_invariant_rows` does densely.
    nkept = length(kept)
    _, schur_vectors, _, left_info = schursolve(
        x -> apply_product(matrix, x; transposed = true), x0, nkept, :LM, arnoldi
    )
    check_converged(left_info, nkept, alg.maxiter)
    left_rows = transpose(stack(schur_vectors[1:nkept]))
    state.iterate = (right_basis, (left_rows * right_basis) \ left_rows, values[kept])
    return state
end

# Arnoldi returns fewer than `howmany` converged values either after `maxiter` restarts or because
# the Krylov space became invariant, in which case the eigenvalues not returned are below `tol`.
function check_converged(info, howmany, maxiter)
    info.converged ≥ howmany || info.numiter < maxiter ||
        error(
        "`KrylovEig` converged $(info.converged) of $howmany eigenvalues in $maxiter iterations."
    )
    return nothing
end

# The dominant invariant subspace of the dense matrix `dense`, truncated by `trunc`.
function dense_subspace(dense, trunc)
    values, right_basis = eig_trunc(dense; trunc)
    left_rows = left_invariant_rows(dense, size(right_basis, 2))
    return right_basis, (left_rows * right_basis) \ left_rows, diag(values)
end
