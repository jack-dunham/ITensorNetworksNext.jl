using Dictionaries: Dictionary, set!
using Graphs: AbstractEdge, dst, src
using ITensorBase: Index, inds, name, names, state
using LinearAlgebra: Diagonal, inv, norm, opnorm
using TensorAlgebra: unmatricize

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
# edge and divided by a corner at each step; it has the bond of the face edge before it.
function right_blocks(embedding, face, transfers, corners, bonds, cut, right_basis)
    last_edge = last(face)
    blocks = Dictionary(
        [last_edge],
        [unmatricize(right_basis, cut, (bonds[prev_edge(embedding, last_edge)],))]
    )
    for edge in reverse(face[2:end])
        previous = prev_edge(embedding, edge)
        block = transfers[edge] * blocks[edge]
        set!(blocks, previous, block * inverse(corners[prev_edge(embedding, previous)]))
    end
    return blocks
end

# Left projector at each edge of `face`, carried forwards from `left_basis` at the last edge
# and divided by a corner at each step; it has the bond of the face edge after it.
function left_blocks(embedding, face, transfers, corners, bonds, cut, left_basis)
    last_edge = last(face)
    blocks = Dictionary(
        [last_edge],
        [
            unmatricize(
                transpose(left_basis),
                cut,
                (bonds[next_edge(embedding, last_edge)],)
            ),
        ]
    )
    for edge in face[1:(end - 1)]
        block = blocks[prev_edge(embedding, edge)] * transfers[edge]
        set!(blocks, edge, inverse(corners[edge]) * block)
    end
    return blocks
end

# Inverse of a diagonal two-index `tensor`, such as a corner, as a map between its indices.
function inverse(tensor)
    row, column = inds(tensor)
    diagonal = [tensor[row => k, column => k] for k in 1:length(row)]
    return diagonal_tensor(inv.(diagonal), row, column)
end

"""
    face_solve(embedding, face, transfers, cut, bonds; maxdim, alg, reference = nothing)
        -> corners, projectors, bases

Solve `face`, given as its cycle of directed edges, from its normalised corner transfer
tensors `transfers` keyed by edge, the indices `cut` of the cut at its last edge, and its
current bond indices `bonds`, which are kept while the bond dimension is unchanged. Reads no
environment, so the transfer tensors can be computed by whichever ranks own the vertices.
`projectors` holds the right and left projectors at each edge of the face, and `bases` the
`(right_basis, left_basis)` of its invariant subspace, which a later solve takes as `reference`
to align onto and, for `SubspaceIteration`, to start from.
"""
function face_solve(
        embedding, face, transfers, cut, bonds;
        maxdim::Integer, alg::AbstractAlgorithm, reference = nothing
    )
    product = CornerTransferProduct(collect(transfers), cut)
    right_basis, left_basis, eigenvalues =
        invariant_subspace(alg, product, maxdim; reference)
    bond_dim = length(eigenvalues)

    # Every corner holds the same `m`-th root of the eigenvalues, which keeps each corner's
    # condition number the `m`-th root of the eigenvalues' instead of concentrating it in one.
    m = length(face)
    roots = if all(value -> isreal(value) && real(value) > 0, eigenvalues)
        real.(eigenvalues) .^ (1 / m)
    else
        complex.(eigenvalues) .^ (1 / m)
    end
    # Ill-conditioned corners or projectors make the inverses below inaccurate.
    @debug(
        "face_solve", face, bond_dim,
        corner_condition = maximum(abs, roots) / minimum(abs, roots),
        projector_condition = opnorm(left_basis) * opnorm(right_basis),
    )

    new_bonds = map(bonds) do old_bond
        return length(old_bond) == bond_dim ? old_bond : Index(bond_dim)
    end
    corners = map(Dictionary(face, face)) do edge
        return diagonal_tensor(roots, new_bonds[edge], new_bonds[next_edge(embedding, edge)])
    end

    # The bases are rescaled so that each pass divides by one corner per step.
    right_projectors = right_blocks(
        embedding, face, transfers, corners, new_bonds, cut,
        right_basis * Diagonal(eigenvalues ./ roots)
    )
    left_projectors = left_blocks(
        embedding, face, transfers, corners, new_bonds, cut,
        Diagonal(inv.(roots)) * left_basis
    )
    return corners, (right_projectors, left_projectors), (right_basis, left_basis)
end

# Writes the edge tensors of `face`'s edges in both directions. It reads the neighbouring
# faces' corners, which must be those the transfer tensors were made with.
function set_face_edge_tensors!(
        env::CTMEnvironment,
        face,
        (right_projectors, left_projectors)
    )
    for edge in face
        reversed = reverse(edge)
        previous = prev_edge(env.embedding, reversed)
        env.edgetensors[reversed] = right_projectors[edge] * inverse(corner(env, reversed))
        env.edgetensors[edge] = left_projectors[edge] * inverse(corner(env, previous))
    end
    return env
end

function set_face_corners!(env::CTMEnvironment, face, corners, bases)
    for edge in face
        env.corners[edge] = corners[edge]
    end
    env.bases[face] = bases
    return env
end

"""
    face_update!(env, tn, face; maxdim, alg) -> env

Replace the corners, bonds and edge tensors of `face`, given as its cycle of directed edges,
with the MP-BP solution of the face given the rest of `env`. The new subspace basis is aligned
onto the one from the face's previous update, so the edge tensors converge entry by entry.
"""
function face_update!(
        env::CTMEnvironment,
        tn,
        face::AbstractVector{<:AbstractEdge};
        maxdim::Integer,
        alg::AbstractAlgorithm
    )
    corners, projectors, bases = face_solve(
        env.embedding, face, transfer_tensors(tn, env, face),
        cut_inds(tn, env, last(face)),
        Dictionary(face, [bond(env, edge) for edge in face]);
        maxdim, alg, reference = get(env.bases, face, nothing)
    )
    set_face_edge_tensors!(env, face, projectors)
    set_face_corners!(env, face, corners, bases)
    return env
end
