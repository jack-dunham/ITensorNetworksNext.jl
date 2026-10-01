using AlgorithmsInterface: AlgorithmsInterface as AI
using Dictionaries: Dictionary, set!
using Graphs: dst, src
using ITensorBase: AbstractNamedTensor, Index, id, name, names, rename, state
using LinearAlgebra: Diagonal, eigen, inv, norm, ordschur, qr, schur
using Random: Xoshiro
using TensorAlgebra: matricize, unmatricize

@kwdef struct DenseEig <: AbstractAlgorithm
    rtol::Float64 = 1.0e-12
    degeneracy_rtol::Float64 = 1.0e-10
end

function invariant_subspace end

function default_algorithm(::typeof(invariant_subspace), ::Type{<:Tuple}; kwargs...)
    return DenseEig(; kwargs...)
end

"""
    TransferProduct(matrices)
    TransferProduct(tensors, cut)

The product `factors[1] * factors[2] * ⋯`, held unformed. Multiplying it by a block from
either side applies the factors one at a time. Named `tensors` are contracted over the cuts
they share and the product maps `cut` to itself; a block's rows or columns then run over `cut`.
"""
struct TransferProduct{F, C <: Tuple}
    factors::Vector{F}
    cut::C
end
TransferProduct(matrices::Vector{<:AbstractMatrix}) = TransferProduct(matrices, ())

const MatrixTransferProduct = TransferProduct{<:AbstractMatrix}
const NamedTransferProduct = TransferProduct{<:AbstractNamedTensor}

Base.eltype(product::TransferProduct) = mapreduce(eltype, promote_type, product.factors)

function Base.size(product::MatrixTransferProduct, dim::Integer)
    return size(dim == 1 ? first(product.factors) : last(product.factors), dim)
end
Base.Matrix(product::MatrixTransferProduct) = foldl(*, product.factors)
function Base.:*(product::MatrixTransferProduct, block::AbstractMatrix)
    return foldr(*, product.factors; init = block)
end
function Base.:*(block::AbstractMatrix, product::MatrixTransferProduct)
    return foldl(*, product.factors; init = block)
end

Base.size(product::NamedTransferProduct, dim::Integer) = prod(length, product.cut)
function Base.Matrix(product::NamedTransferProduct)
    # The first factor's copy of `cut` is renamed, so the last factor does not contract with it.
    output = map(index -> Index(length(index)), product.cut)
    first_factor = rename(first(product.factors), (product.cut .=> output)...)
    closed = foldl(*, product.factors[2:end]; init = first_factor)
    return matricize(closed, output, product.cut)
end
function Base.:*(product::NamedTransferProduct, block::AbstractMatrix)
    column = Index(size(block, 2))
    named_block = unmatricize(block, product.cut, (column,))
    return matricize(foldr(*, product.factors; init = named_block), product.cut, (column,))
end
function Base.:*(block::AbstractMatrix, product::NamedTransferProduct)
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

"""
    invariant_subspace(alg::DenseEig, matrix, maxdim) -> (right_basis, left_basis, eigenvalues)

Dominant invariant subspace of `matrix`: `matrix * right_basis ≈ right_basis * Diagonal(eigenvalues)`,
`left_basis * matrix ≈ Diagonal(eigenvalues) * left_basis` and `left_basis * right_basis ≈ I`.
Eigenvalues of equal modulus are kept or dropped together.
"""
function invariant_subspace(alg::DenseEig, matrix, maxdim::Integer)
    dense = Matrix(matrix)
    decomposition = eigen(dense)
    order = sortperm(abs.(decomposition.values); rev = true)
    eigenvalues, eigenvectors = decomposition.values[order], decomposition.vectors[:, order]
    nkept = kept_count(alg, eigenvalues, maxdim)

    right_basis = eigenvectors[:, 1:nkept]
    left_rows = left_invariant_rows(dense, eigenvalues, nkept)
    return right_basis, (left_rows * right_basis) \ left_rows, eigenvalues[1:nkept]
end

"""
    SubspaceIteration(; oversampling = 2, tol = 1.0e-12, maxiter = 1000, seed = 0, rtol, degeneracy_rtol)

Two-sided block subspace iteration with `maxdim + oversampling` vectors, started from random
blocks drawn with `seed`. It reads `matrix` only through `matrix * block` and `block * matrix`,
so `matrix` can be a `TransferProduct`. It stops once the kept Ritz pairs have relative
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

struct InvariantSubspaceProblem{Operator} <: AI.Problem
    matrix::Operator
    maxdim::Int
end

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

function invariant_subspace(alg::SubspaceIteration, matrix, maxdim::Integer)
    dim = size(matrix, 1)
    nblock = min(dim, maxdim + alg.oversampling)
    rng = Xoshiro(alg.seed)
    right = orthonormal_columns(randn(rng, eltype(matrix), dim, nblock))
    left = transpose(orthonormal_columns(randn(rng, eltype(matrix), dim, nblock)))

    problem = InvariantSubspaceProblem(matrix, maxdim)
    stopping_criterion = AI.StopAfterIteration(alg.maxiter) | StopWhenResidualBelow(alg.tol)
    algorithm = RayleighRitzIteration(alg, stopping_criterion)
    return AI.solve(problem, algorithm; iterate = (right, left))
end

cut_inds(tn, env::CTMEnvironment, edge) = (linkinds(tn, edge)..., bond(env, reverse(edge)))

# Transfer matrix at `src(outgoing)` from the face edge before `outgoing` to `outgoing`.
function corner_transfer_matrix(tn, env::CTMEnvironment, outgoing)
    incoming = prev_edge(env.embedding, outgoing)
    vertex = src(outgoing)
    exclude = (src(incoming), dst(outgoing))

    transfer = contract_network([[tn[vertex]]; environment_tensors(env, vertex; exclude)])

    for reversed in (reverse(incoming), reverse(outgoing))
        bond_index = bond(env, reversed)
        if name(bond_index) ∉ names(transfer)
            @assert length(bond_index) == 1
            transfer = transfer * ones(eltype(transfer), (bond_index,))
        end
    end

    @assert issetequal(
        names(transfer),
        name.((cut_inds(tn, env, incoming)..., cut_inds(tn, env, outgoing)...))
    )
    return transfer
end

# Normalised transfer tensor at each edge of `face`.
function transfer_tensors(tn, env::CTMEnvironment, face::Int)
    face_edges = env.embedding.faces[face]
    transfers = map(face_edges) do edge
        # `state` drops the operator pairing a `NormNetwork` vertex tensor carries.
        transfer = state(corner_transfer_matrix(tn, env, edge))
        return transfer / norm(transfer)
    end
    return Dictionary(face_edges, transfers)
end

# The eigenvalue corner of a face sits between this edge and the face's last edge.
eigenvalue_edge(face_edges) = face_edges[end - 1]

# Right projector at each edge of `face`, carried backwards from `right_basis` at the last
# edge; the one at an edge has that edge's cut and the bond of the face edge before it.
function right_blocks(tn, env::CTMEnvironment, face::Int, transfers, right_basis)
    embedding = env.embedding
    face_edges = embedding.faces[face]
    last_edge = last(face_edges)
    bond_before(edge) = bond(env, prev_edge(embedding, edge))

    blocks = Dictionary(
        [last_edge],
        [unmatricize(right_basis, cut_inds(tn, env, last_edge), (bond_before(last_edge),))]
    )
    for edge in reverse(face_edges[2:end])
        previous = prev_edge(embedding, edge)
        block = transfers[edge] * blocks[edge]
        set!(blocks, previous, rename(block, bond_before(edge) => bond_before(previous)))
    end
    return blocks
end

# Left projector at each edge of `face`, carried forwards from `left_basis` at the last edge
# and divided by `eigenvalue_corner` at its edge; it has the bond of the face edge after it.
function left_blocks(
        tn,
        env::CTMEnvironment,
        face::Int,
        transfers,
        left_basis,
        eigenvalue_corner
    )
    embedding = env.embedding
    face_edges = embedding.faces[face]
    last_edge = last(face_edges)
    divided_edge = eigenvalue_edge(face_edges)
    bond_after(edge) = bond(env, next_edge(embedding, edge))

    blocks = Dictionary(
        [last_edge],
        [
            unmatricize(
                transpose(left_basis), cut_inds(tn, env, last_edge),
                (bond_after(last_edge),)
            ),
        ]
    )
    inverse_corner = unmatricize(
        inv(eigenvalue_corner), (bond_after(divided_edge),), (bond(env, divided_edge),)
    )
    for edge in face_edges[1:(end - 1)]
        block = blocks[prev_edge(embedding, edge)] * transfers[edge]
        block = if edge == divided_edge
            inverse_corner * block
        else
            rename(block, bond(env, edge) => bond_after(edge))
        end
        set!(blocks, edge, block)
    end
    return blocks
end

# `product` equals `A` contracted with `corner_tensor` over `new_leg`; returns `A`, whose leg
# `old_leg` becomes `new_leg`.
function peel(product, corner_tensor, old_leg, new_leg)
    inverse = inv(matricize(corner_tensor, (new_leg,), (old_leg,)))
    return product * unmatricize(inverse, (old_leg,), (new_leg,))
end

"""
    face_update!(env, tn, face; maxdim, alg) -> env

Replace the corners, bonds and edge tensors of `face` with the MP-BP solution of the face
given the rest of `env`.
"""
function face_update!(
        env::CTMEnvironment,
        tn,
        face::Int;
        maxdim::Integer,
        alg::AbstractAlgorithm
    )
    embedding = env.embedding
    face_edges = embedding.faces[face]
    nedges = length(face_edges)
    if nedges < 3
        throw(
            ArgumentError("Face $face has $nedges edges; `face_update!` needs at least 3.")
        )
    end

    transfers = transfer_tensors(tn, env, face)
    product = TransferProduct(collect(transfers), cut_inds(tn, env, last(face_edges)))

    right_basis, left_basis, eigenvalues = invariant_subspace(alg, product, maxdim)
    bond_dim = length(eigenvalues)
    eigenvalue_corner = Matrix(Diagonal(eigenvalues))

    # The projectors and corners below take the face's new bonds from `env`.
    foreach(edge -> set!(env.bonds, edge, Index(bond_dim)), face_edges)
    right_projectors = right_blocks(tn, env, face, transfers, right_basis)
    left_projectors = left_blocks(tn, env, face, transfers, left_basis, eigenvalue_corner)

    for edge in face_edges
        reversed = reverse(edge)

        env.edgetensors[reversed] = peel(
            right_projectors[edge], corner(env, reversed),
            bond(env, reversed), bond(env, next_edge(embedding, reversed))
        )
        before_reversed = prev_edge(embedding, reversed)
        env.edgetensors[edge] = peel(
            left_projectors[edge], corner(env, before_reversed),
            bond(env, reversed), bond(env, before_reversed)
        )
    end

    elt = promote_type(eltype(right_basis), eltype(left_basis), eltype(eigenvalue_corner))
    for edge in face_edges
        bonds = ((bond(env, edge),), (bond(env, next_edge(embedding, edge)),))
        env.corners[edge] = if edge == eigenvalue_edge(face_edges)
            unmatricize(Matrix{elt}(eigenvalue_corner), bonds...)
        else
            id(elt, bonds...)
        end
    end
    return env
end
