using AlgorithmsInterface: AlgorithmsInterface as AI
using ITensorBase: AbstractNamedTensor, Index, rename
using LinearAlgebra: Diagonal, cond, eigen, norm, ordschur, qr, schur
using Random: Xoshiro
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
    CornerTransferProduct(matrices)
    CornerTransferProduct(tensors, cut)

The product `factors[1] * factors[2] * ⋯`, held unformed. Multiplying it by a block from
either side applies the factors one at a time. Named `tensors` are contracted over the cuts
they share and the product maps `cut` to itself; a block's rows or columns then run over `cut`.
"""
struct CornerTransferProduct{F, C <: Tuple}
    factors::Vector{F}
    cut::C
end
function CornerTransferProduct(matrices::Vector{<:AbstractMatrix})
    return CornerTransferProduct(matrices, ())
end

const MatrixCornerTransferProduct = CornerTransferProduct{<:AbstractMatrix}
const NamedCornerTransferProduct = CornerTransferProduct{<:AbstractNamedTensor}

function Base.eltype(product::CornerTransferProduct)
    return mapreduce(eltype, promote_type, product.factors)
end

function Base.size(product::MatrixCornerTransferProduct, dim::Integer)
    return size(dim == 1 ? first(product.factors) : last(product.factors), dim)
end
Base.Matrix(product::MatrixCornerTransferProduct) = foldl(*, product.factors)
function Base.:*(product::MatrixCornerTransferProduct, block::AbstractMatrix)
    return foldr(*, product.factors; init = block)
end
function Base.:*(block::AbstractMatrix, product::MatrixCornerTransferProduct)
    return foldl(*, product.factors; init = block)
end

Base.size(product::NamedCornerTransferProduct, dim::Integer) = prod(length, product.cut)
function Base.Matrix(product::NamedCornerTransferProduct)
    # The first factor's copy of `cut` is renamed, so the last factor does not contract with it.
    output = map(index -> Index(length(index)), product.cut)
    first_factor = rename(first(product.factors), (product.cut .=> output)...)
    closed = foldl(*, product.factors[2:end]; init = first_factor)
    return matricize(closed, output, product.cut)
end
function Base.:*(product::NamedCornerTransferProduct, block::AbstractMatrix)
    column = Index(size(block, 2))
    named_block = unmatricize(block, product.cut, (column,))
    return matricize(foldr(*, product.factors; init = named_block), product.cut, (column,))
end
function Base.:*(block::AbstractMatrix, product::NamedCornerTransferProduct)
    row = Index(size(block, 1))
    named_block = unmatricize(block, (row,), product.cut)
    return matricize(foldl(*, product.factors; init = named_block), (row,), product.cut)
end

# Number of leading `eigenvalues`, sorted by decreasing modulus, to keep: at most `maxdim`, above
# `alg.rtol` relative to the largest, and never splitting eigenvalues of equal modulus.
function kept_count(alg, eigenvalues, maxdim)
    scale = abs(first(eigenvalues))
    nkept = min(maxdim, count(value -> abs(value) > alg.rtol * scale, eigenvalues))
    while 0 < nkept < length(eigenvalues) &&
            abs(abs(eigenvalues[nkept]) - abs(eigenvalues[nkept + 1])) ≤
            alg.degeneracy_rtol * abs(eigenvalues[nkept])
        nkept -= 1
    end
    nkept > 0 || throw(
        ArgumentError(
            "`maxdim = $maxdim` splits the dominant eigenvalue multiplet; raise `maxdim`."
        )
    )
    return nkept
end

# Rows spanning the left invariant subspace of `matrix` for its `nkept` eigenvalues of largest
# modulus, from an ordered Schur form, so that no eigenvectors of the discarded ones enter.
function left_invariant_rows(matrix, eigenvalues, nkept)
    threshold = if nkept < length(eigenvalues)
        (abs(eigenvalues[nkept]) + abs(eigenvalues[nkept + 1])) / 2
    else
        -one(real(eltype(eigenvalues)))
    end
    decomposition = schur(Matrix(transpose(matrix)))
    decomposition = ordschur(decomposition, abs.(decomposition.values) .> threshold)
    return transpose(decomposition.Z[:, 1:nkept])
end

# Matches each new eigenvector to the previous basis vector it overlaps most, then rotates the
# bases onto `previous` within blocks of equal eigenvalues, which leaves the corners diagonal.
function align_bases(right_basis, left_basis, eigenvalues, previous, degeneracy_rtol)
    size(previous) == size(right_basis) || return right_basis, left_basis, eigenvalues
    overlaps = abs.(left_basis * previous)
    order = Int[]
    for column in axes(overlaps, 2)
        candidates = setdiff(axes(overlaps, 1), order)
        push!(order, candidates[argmax(overlaps[candidates, column])])
    end
    right_basis, left_basis = right_basis[:, order], left_basis[order, :]
    eigenvalues = eigenvalues[order]

    rotation = left_basis * previous
    for i in axes(rotation, 1), j in axes(rotation, 2)
        degenerate =
            i == j ||
            abs(eigenvalues[i] - eigenvalues[j]) ≤ degeneracy_rtol * abs(eigenvalues[i])
        degenerate || (rotation[i, j] = 0)
    end
    # A near-singular rotation means the subspace has moved; aligning onto it would amplify noise.
    cond(rotation) < 1.0e8 || return right_basis, left_basis, eigenvalues
    return right_basis * rotation, rotation \ left_basis, eigenvalues
end

"""
    invariant_subspace(alg::DenseEig, matrix, maxdim; reference = nothing)
        -> (right_basis, left_basis, eigenvalues)

Dominant invariant subspace of `matrix`: `matrix * right_basis ≈ right_basis * Diagonal(eigenvalues)`,
`left_basis * matrix ≈ Diagonal(eigenvalues) * left_basis` and `left_basis * right_basis ≈ I`.
Eigenvalues of equal modulus are kept or dropped together.

With `reference = (right_basis, left_basis)` from a previous call of the same size, the right
basis is rotated onto the reference's within blocks of eigenvalues equal to relative tolerance
`alg.degeneracy_rtol`, and the eigenvalues follow the reference's order.
"""
function invariant_subspace(alg::DenseEig, matrix, maxdim::Integer; reference = nothing)
    result = AI.solve(InvariantSubspaceProblem(matrix, maxdim), DenseEigensolve(alg))
    return aligned(result, reference, alg)
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
    decomposition = eigen(dense)
    order = sortperm(abs.(decomposition.values); rev = true)
    eigenvalues, eigenvectors = decomposition.values[order], decomposition.vectors[:, order]
    nkept = kept_count(algorithm.dense_eig, eigenvalues, maxdim)

    right_basis = eigenvectors[:, 1:nkept]
    left_rows = left_invariant_rows(dense, eigenvalues, nkept)
    state.iterate =
        (right_basis, (left_rows * right_basis) \ left_rows, eigenvalues[1:nkept])
    return state
end

"""
    SubspaceIteration(; oversampling = 2, tol = 1.0e-12, maxiter = 1000, seed = 0, rtol, degeneracy_rtol)

Two-sided block subspace iteration with `maxdim + oversampling` vectors, started from a
`reference` pair of bases when one is given, and otherwise from random blocks drawn with `seed`. It reads `matrix` only through `matrix * block` and `block * matrix`,
so `matrix` can be a `CornerTransferProduct`. It stops once the kept Ritz pairs have relative
residual below `tol` on both sides, and throws after `maxiter` iterations otherwise.
"""
@kwdef struct SubspaceIteration <: AbstractAlgorithm
    oversampling::Int = 2
    tol::Float64 = 1.0e-12
    maxiter::Int = 1000
    seed::Int = 0
    # An eigenvector this close to rounding changes with the starting blocks, so each call moves the
    # neighbouring faces.
    rtol::Float64 = 1.0e-9
    degeneracy_rtol::Float64 = 1.0e-10
end

orthonormal_columns(block) = Matrix(qr(block).Q)

# Iterates `subspace_iteration`'s Rayleigh–Ritz step; the iterate is the (right, left) pair of
# blocks, and `ritz` holds the kept Ritz pairs of the last step.
struct RayleighRitzIteration{StoppingCriterion <: AI.StoppingCriterion} <: AI.Algorithm
    subspace_iteration::SubspaceIteration
    stopping_criterion::StoppingCriterion
end

@kwdef mutable struct RayleighRitzState{Iterate, StoppingCriterionState} <: AI.State
    iterate::Iterate
    iteration::Int = 0
    ritz = nothing
    residual::Float64 = Inf
    stopping_criterion_state::StoppingCriterionState
end

function AI.initialize_state(
        problem::InvariantSubspaceProblem, algorithm::RayleighRitzIteration; iterate
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return RayleighRitzState(; iterate, stopping_criterion_state)
end

function AI.step!(
        problem::InvariantSubspaceProblem, algorithm::RayleighRitzIteration,
        state::RayleighRitzState
    )
    (; matrix, maxdim) = problem
    right, left = state.iterate
    right_image = matrix * right
    left_image = left * matrix

    overlap = left * right
    ritz_matrix = overlap \ (left * right_image)
    decomposition = eigen(ritz_matrix)
    order = sortperm(abs.(decomposition.values); rev = true)
    eigenvalues = decomposition.values[order]
    ritz_vectors = decomposition.vectors[:, order]
    nkept = kept_count(algorithm.subspace_iteration, eigenvalues, maxdim)

    kept = 1:nkept
    right_basis = right * ritz_vectors[:, kept]
    left_rows = left_invariant_rows(ritz_matrix, eigenvalues, nkept)
    left_transform = (left_rows * (overlap \ left) * right_basis) \ left_rows
    left_basis = left_transform * (overlap \ left)
    values = Diagonal(eigenvalues[kept])
    scale = abs(first(eigenvalues))
    right_residual =
        norm(right_image * ritz_vectors[:, kept] - right_basis * values) /
        (scale * norm(right_basis))
    left_residual =
        norm(left_transform * (overlap \ left_image) - values * left_basis) /
        (scale * norm(left_basis))

    state.ritz = (right_basis, left_basis, eigenvalues[kept])
    state.residual = max(right_residual, left_residual)
    state.iterate = (
        orthonormal_columns(right_image),
        transpose(orthonormal_columns(transpose(left_image))),
    )
    return state
end

function AI.finalize_state!(
        ::InvariantSubspaceProblem, algorithm::RayleighRitzIteration, state::RayleighRitzState
    )
    (; tol, maxiter) = algorithm.subspace_iteration
    state.residual < tol || error(
        "`SubspaceIteration` stopped after $maxiter iterations with residual " *
            "$(state.residual), not below `tol` = $tol."
    )
    return state.ritz
end

# Stops once the `residual` of a `RayleighRitzState` is below `tol`.
struct StopWhenResidualBelow <: AI.StoppingCriterion
    tol::Float64
end

function AI.initialize_state(
        ::AI.Problem, ::AI.Algorithm, ::StopWhenResidualBelow; kwargs...
    )
    return AI.DefaultStoppingCriterionState()
end
function AI.initialize_state!(
        ::AI.Problem, ::AI.Algorithm, ::StopWhenResidualBelow,
        criterion_state::AI.DefaultStoppingCriterionState; kwargs...
    )
    criterion_state.at_iteration = -1
    return criterion_state
end
function AI.is_finished(
        ::AI.Problem, ::AI.Algorithm, state::AI.State, criterion::StopWhenResidualBelow,
        ::AI.DefaultStoppingCriterionState
    )
    return state.residual < criterion.tol
end
function AI.is_finished!(
        problem::AI.Problem, algorithm::AI.Algorithm, state::AI.State,
        criterion::StopWhenResidualBelow, criterion_state::AI.DefaultStoppingCriterionState
    )
    finished = AI.is_finished(problem, algorithm, state, criterion, criterion_state)
    finished && (criterion_state.at_iteration = state.iteration)
    return finished
end
AI.indicates_convergence(::StopWhenResidualBelow) = true

function invariant_subspace(
        alg::SubspaceIteration,
        matrix,
        maxdim::Integer;
        reference = nothing
    )
    dim = size(matrix, 1)
    nblock = min(dim, maxdim + alg.oversampling)
    rng = Xoshiro(alg.seed)
    # A reference basis is padded with random columns up to the block size.
    function start(columns)
        filler = randn(rng, eltype(matrix), dim, nblock - min(nblock, size(columns, 2)))
        return orthonormal_columns(hcat(columns[:, 1:min(nblock, end)], filler))
    end
    usable = !isnothing(reference) && size(first(reference), 1) == dim
    right = start(usable ? first(reference) : zeros(eltype(matrix), dim, 0))
    left = transpose(
        start(usable ? transpose(last(reference)) : zeros(eltype(matrix), dim, 0))
    )

    problem = InvariantSubspaceProblem(matrix, maxdim)
    stopping_criterion = AI.StopAfterIteration(alg.maxiter) | StopWhenResidualBelow(alg.tol)
    algorithm = RayleighRitzIteration(alg, stopping_criterion)
    result = AI.solve(problem, algorithm; iterate = (right, left))
    return aligned(result, reference, alg)
end
