using Dictionaries: Dictionary, set!
using Graphs: AbstractEdge, dst, src
using ITensorBase: Index, inds, name, names, rename, state
using LinearAlgebra: inv, norm, opnorm
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
function transfer_tensors(tn, env::CTMEnvironment, face::AbstractVector{<:AbstractEdge})
    transfers = map(face) do edge
        # `state` drops the operator pairing a `NormNetwork` vertex tensor carries.
        transfer = state(corner_transfer_matrix(tn, env, edge))
        return transfer / norm(transfer)
    end
    return Dictionary(face, transfers)
end

# Right projector at each edge of `face`, carried backwards from `right_basis` at the last
# edge; the one at an edge has that edge's cut and the bond of the face edge before it.
function right_blocks(
        tn,
        env::CTMEnvironment,
        face::AbstractVector{<:AbstractEdge},
        transfers,
        right_basis
    )
    embedding = env.embedding
    last_edge = last(face)
    bond_before(edge) = bond(env, prev_edge(embedding, edge))

    blocks = Dictionary(
        [last_edge],
        [unmatricize(right_basis, cut_inds(tn, env, last_edge), (bond_before(last_edge),))]
    )
    for edge in reverse(face[2:end])
        previous = prev_edge(embedding, edge)
        block = transfers[edge] * blocks[edge]
        set!(blocks, previous, rename(block, bond_before(edge) => bond_before(previous)))
    end
    return blocks
end

# Left projector at each edge of `face`, carried forwards from `left_basis` at the last edge
# and divided by the face's eigenvalue corner; it has the bond of the face edge after it.
function left_blocks(
        tn,
        env::CTMEnvironment,
        face::AbstractVector{<:AbstractEdge},
        transfers,
        left_basis
    )
    embedding = env.embedding
    last_edge = last(face)
    divided_edge = face[end - 1]
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
    for edge in face[1:(end - 1)]
        block = blocks[prev_edge(embedding, edge)] * transfers[edge]
        block = if edge == divided_edge
            inverse(corner(env, divided_edge)) * block
        else
            rename(block, bond(env, edge) => bond_after(edge))
        end
        set!(blocks, edge, block)
    end
    return blocks
end

# Inverse of a two-index `tensor` as a map between its indices, so contracting a tensor with
# `tensor` and then with this leaves it unchanged.
function inverse(tensor)
    rows, columns = inds(tensor)
    return unmatricize(inv(matricize(tensor, (rows,), (columns,))), (columns,), (rows,))
end

"""
    face_update!(env, tn, face; maxdim, alg) -> env

Replace the corners, bonds and edge tensors of `face`, given as its cycle of directed edges,
with the MP-BP solution of the face given the rest of `env`.
"""
function face_update!(
        env::CTMEnvironment,
        tn,
        face::AbstractVector{<:AbstractEdge};
        maxdim::Integer,
        alg::AbstractAlgorithm
    )
    embedding = env.embedding

    transfers = transfer_tensors(tn, env, face)
    product = CornerTransferProduct(collect(transfers), cut_inds(tn, env, last(face)))

    right_basis, left_basis, eigenvalues = invariant_subspace(alg, product, maxdim)
    bond_dim = length(eigenvalues)
    # Ill-conditioned eigenvalue corners or projectors make the inverses below inaccurate.
    @debug(
        "face_update!", face, bond_dim,
        corner_condition = maximum(abs, eigenvalues) / minimum(abs, eigenvalues),
        projector_condition = opnorm(left_basis) * opnorm(right_basis),
    )

    # A face's bonds are read off its corners, so all of them are replaced before the
    # projectors read them. Every corner is the identity except the eigenvalue edge's.
    new_bonds = Dictionary(face, [Index(bond_dim) for _ in face])

    for edge in face
        data = edge == face[end - 1] ? eigenvalues : one.(eigenvalues)

        row, column = new_bonds[edge], new_bonds[next_edge(embedding, edge)]

        tensor = zeros(eltype(data), (row, column))

        for (k, value) in enumerate(data)
            tensor[row => k, column => k] = value
        end

        env.corners[edge] = tensor
    end

    right_projectors = right_blocks(tn, env, face, transfers, right_basis)
    left_projectors = left_blocks(tn, env, face, transfers, left_basis)
    for edge in face
        reversed = reverse(edge)
        previous = prev_edge(embedding, reversed)
        env.edgetensors[reversed] = right_projectors[edge] * inverse(corner(env, reversed))
        env.edgetensors[edge] = left_projectors[edge] * inverse(corner(env, previous))
    end
    return env
end
