using Dictionaries: set!
using Graphs: dst, src
using ITensorBase: Index, name, names
using LinearAlgebra: Diagonal, I, eigen, inv, norm, ordschur, qr, schur
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

The product `matrices[1] * matrices[2] * ⋯`, held unformed. Multiplying it by a block from
either side applies the factors one at a time.
"""
struct TransferProduct{M <: AbstractMatrix}
    matrices::Vector{M}
end

function Base.size(product::TransferProduct, dim::Integer)
    return size(dim == 1 ? first(product.matrices) : last(product.matrices), dim)
end
Base.eltype(product::TransferProduct) = mapreduce(eltype, promote_type, product.matrices)
Base.Matrix(product::TransferProduct) = foldl(*, product.matrices)
function Base.:*(product::TransferProduct, block::AbstractMatrix)
    return foldr(*, product.matrices; init = block)
end
function Base.:*(block::AbstractMatrix, product::TransferProduct)
    return foldl(*, product.matrices; init = block)
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

function invariant_subspace(alg::SubspaceIteration, matrix, maxdim::Integer)
    dim = size(matrix, 1)
    nblock = min(dim, maxdim + alg.oversampling)
    rng = Xoshiro(alg.seed)
    right = orthonormal_columns(randn(rng, eltype(matrix), dim, nblock))
    left = transpose(orthonormal_columns(randn(rng, eltype(matrix), dim, nblock)))

    residual = Inf
    for _ in 1:alg.maxiter
        right_image = matrix * right
        left_image = left * matrix

        # Rayleigh–Ritz on the pair of blocks.
        overlap = left * right
        ritz_matrix = overlap \ (left * right_image)
        decomposition = eigen(ritz_matrix)
        order = sortperm(abs.(decomposition.values); rev = true)
        eigenvalues = decomposition.values[order]
        ritz_vectors = decomposition.vectors[:, order]
        nkept = kept_count(alg, eigenvalues, maxdim)

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
        residual = max(right_residual, left_residual)
        residual < alg.tol && return right_basis, left_basis, eigenvalues[kept]

        right = orthonormal_columns(right_image)
        left = transpose(orthonormal_columns(transpose(left_image)))
    end
    return error(
        "`SubspaceIteration` stopped after $(alg.maxiter) iterations with residual " *
            "$residual, not below `tol` = $(alg.tol)."
    )
end

cut_inds(tn, env::CTMEnvironment, edge) = (linkinds(tn, edge)..., bond(env, reverse(edge)))

function corner_transfer_matrix(tn, env::CTMEnvironment, face::Int, position::Int)
    face_edges = env.embedding.faces[face]
    incoming = face_edges[mod1(position - 1, length(face_edges))]
    outgoing = face_edges[position]
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

# `product` equals `A` contracted with `corner_tensor` over `new_leg`; returns `A`, whose leg
# `old_leg` becomes `new_leg`.
function peel(product, corner_tensor, old_leg, new_leg)
    inverse = inv(matricize(corner_tensor, (new_leg,), (old_leg,)))
    return product * unmatricize(inverse, (old_leg,), (new_leg,))
end

"""
    face_update!(env, tn, face; maxdim, alg, frozen = false, align = false) -> env

Replace the corners, bonds and edge tensors of `face` with the MP-BP solution of the face
given the rest of `env`.

With `frozen = true` the face keeps its bond indices, so the kept subspace must have their
dimension. With `align = true` the new subspace basis is rotated onto the one the previous
update of `face` stored, which keeps the tensors continuous between updates when eigenvalues
share a modulus; the eigenvalue corner is then a full matrix.
"""
function face_update!(
        env::CTMEnvironment,
        tn,
        face::Int;
        maxdim::Integer,
        alg::AbstractAlgorithm,
        frozen::Bool = false,
        align::Bool = false
    )
    embedding = env.embedding
    face_edges = embedding.faces[face]
    nedges = length(face_edges)
    if nedges < 3
        throw(
            ArgumentError("Face $face has $nedges edges; `face_update!` needs at least 3.")
        )
    end

    cut_indices = [cut_inds(tn, env, edge) for edge in face_edges]
    transfer_tensors = [corner_transfer_matrix(tn, env, face, i) for i in 1:nedges]
    transfer_matrices = [
        matricize(
                transfer_tensors[i] / norm(transfer_tensors[i]),
                cut_indices[mod1(i - 1, nedges)],
                cut_indices[i]
            ) for i in 1:nedges
    ]

    frozen_dim = length(bond(env, first(face_edges)))
    right_basis, left_basis, eigenvalues = invariant_subspace(
        alg, TransferProduct(transfer_matrices), frozen ? frozen_dim : maxdim
    )
    bond_dim = length(eigenvalues)
    if frozen && bond_dim != frozen_dim
        throw(
            ArgumentError(
                "Face $face keeps $bond_dim eigenvalues but its frozen bonds have dimension $frozen_dim."
            )
        )
    end

    eigenvalue_corner = Matrix(Diagonal(eigenvalues))
    if align
        previous_basis = get(env.gauges, face, nothing)
        if !isnothing(previous_basis) && size(previous_basis) == size(right_basis)
            rotation = left_basis * previous_basis
            right_basis = right_basis * rotation
            left_basis = rotation \ left_basis
            eigenvalue_corner = rotation \ (eigenvalue_corner * rotation)
        end
        env.gauges[face] = right_basis
    end

    right_bases = Vector{Matrix{eltype(right_basis)}}(undef, nedges)
    left_bases = Vector{Matrix{eltype(left_basis)}}(undef, nedges)
    right_bases[nedges], left_bases[nedges] = right_basis, left_basis
    for i in nedges:-1:2
        right_bases[i - 1] = transfer_matrices[i] * right_bases[i]
    end

    left_bases[1] = left_bases[nedges] * transfer_matrices[1]
    for i in 2:(nedges - 2)
        left_bases[i] = left_bases[i - 1] * transfer_matrices[i]
    end
    left_bases[nedges - 1] =
        (eigenvalue_corner \ left_bases[nedges - 2]) * transfer_matrices[nedges - 1]

    new_bonds = if frozen
        [bond(env, edge) for edge in face_edges]
    else
        [Index(bond_dim) for _ in 1:nedges]
    end

    for i in 1:nedges
        edge = face_edges[i]
        reversed = reverse(edge)

        right_projector =
            unmatricize(right_bases[i], cut_indices[i], (new_bonds[mod1(i - 1, nedges)],))
        left_projector = unmatricize(
            transpose(left_bases[i]), cut_indices[i], (new_bonds[mod1(i + 1, nedges)],)
        )

        env.edgetensors[reversed] = peel(
            right_projector, corner(env, reversed),
            bond(env, reversed), bond(env, next_edge(embedding, reversed))
        )
        before_reversed = prev_edge(embedding, reversed)
        env.edgetensors[edge] = peel(
            left_projector, corner(env, before_reversed),
            bond(env, reversed), bond(env, before_reversed)
        )
    end

    elt = promote_type(eltype(right_basis), eltype(left_basis), eltype(eigenvalue_corner))
    for i in 1:nedges
        corner_matrix = if i == nedges - 1
            Matrix{elt}(eigenvalue_corner)
        else
            Matrix{elt}(I, bond_dim, bond_dim)
        end
        env.corners[face_edges[i]] =
            unmatricize(corner_matrix, (new_bonds[i],), (new_bonds[mod1(i + 1, nedges)],))
        frozen || set!(env.bonds, face_edges[i], new_bonds[i])
    end
    return env
end
