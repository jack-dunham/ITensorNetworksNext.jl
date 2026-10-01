using Dictionaries: Dictionary, set!
using Graphs: dst, src
using ITensorBase: Index, id, name, names, rename, state
using LinearAlgebra: Diagonal, inv, norm
using TensorAlgebra: matricize, unmatricize

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
