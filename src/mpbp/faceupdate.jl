using Dictionaries: Dictionary, set!
using Graphs: dst, src
using ITensorBase: Index, commoninds, inds, name, names, state
using LinearAlgebra: Diagonal, inv, norm, opnorm
using TensorAlgebra: unmatricize

cut_inds(tn, env::CTMEnvironment, edge) = (linkinds(tn, edge)..., bond(env, reverse(edge)))

# Transfer matrix at `src(outgoing)` from the face edge before `outgoing` to `outgoing`.
function corner_transfer_matrix(tn, env::CTMEnvironment, outgoing)
    incoming = prevedge(env.embedding, outgoing)
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
function transfer_tensors(tn, env::CTMEnvironment, face::NamedFace)
    transfers = map(face) do edge
        # `state` drops the operator pairing a `NormNetwork` vertex tensor carries.
        transfer = state(corner_transfer_matrix(tn, env, edge))
        return transfer / norm(transfer)
    end
    return Dictionary(face, transfers)
end

# Right projector at each edge of `face`, carried backwards from `right_basis` at the last
# edge and divided by a corner at each step; it has the bond of the face edge before it.
# The cut at `face`'s last edge: the indices its last and first transfer tensors share.
face_cut(face, transfers) = Tuple(commoninds(transfers[last(face)], transfers[first(face)]))

function right_blocks(env::CTMEnvironment, face, transfers, right_basis)
    embedding = env.embedding
    last_edge = last(face)
    blocks = Dictionary(
        [last_edge],
        [
            unmatricize(
                right_basis, face_cut(face, transfers),
                (bond(env, prevedge(embedding, last_edge)),)
            ),
        ]
    )
    for edge in reverse(face[2:end])
        previous = prevedge(embedding, edge)
        block = transfers[edge] * blocks[edge]
        set!(
            blocks,
            previous,
            block * inverse(cornertensor(env, prevedge(embedding, previous)))
        )
    end
    return blocks
end

# Left projector at each edge of `face`, carried forwards from `left_basis` at the last edge
# and divided by a corner at each step; it has the bond of the face edge after it.
function left_blocks(env::CTMEnvironment, face, transfers, left_basis)
    embedding = env.embedding
    last_edge = last(face)
    blocks = Dictionary(
        [last_edge],
        [
            unmatricize(
                transpose(left_basis),
                face_cut(face, transfers),
                (bond(env, nextedge(embedding, last_edge)),)
            ),
        ]
    )
    for edge in face[1:(end - 1)]
        block = blocks[prevedge(embedding, edge)] * transfers[edge]
        set!(blocks, edge, inverse(cornertensor(env, edge)) * block)
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
    face_solve(embedding, face, transfers, bonds; maxdim, alg, reference = nothing)
        -> corners, bases

Solve the `NamedFace` `face` from its normalised corner transfer
tensors `transfers` keyed by edge, and its current bond indices `bonds`, which are kept while the bond dimension is unchanged. Reads no
environment, so the transfer tensors can be computed by whichever ranks own the vertices.
`bases` is the `(right_basis, left_basis)` of the face's invariant subspace, from which
`set_face_edges!` builds the projectors and which a later solve takes as `reference`.
"""
function face_solve(
        embedding, face, transfers, bonds;
        maxdim::Integer, alg::AbstractAlgorithm, reference = nothing
    )
    product = CornerTransferProduct(collect(transfers), face_cut(face, transfers))
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
    # Ill-conditioned corners or bases make the edge tensors written from them inaccurate.
    @debug(
        "face_solve", face, bond_dim,
        corner_condition = maximum(abs, roots) / minimum(abs, roots),
        projector_condition = opnorm(left_basis) * opnorm(right_basis),
    )

    new_bonds = map(bonds) do old_bond
        return length(old_bond) == bond_dim ? old_bond : Index(bond_dim)
    end
    corners = map(Dictionary(face, face)) do edge
        return diagonal_tensor(roots, new_bonds[edge], new_bonds[nextedge(embedding, edge)])
    end
    return corners, (right_basis, left_basis)
end

function set_face_corners!(env::CTMEnvironment, face, corners)
    for edge in face
        env.cornertensors[edge] = corners[edge]
    end
    return env
end

set_face_bases!(env::CTMEnvironment, face, bases) = (env.bases[face] = bases; env)

"""
    set_face_edges!(env, face, transfers, (right_basis, left_basis)) -> env

Build the projectors at each edge of `face` from its bases and write the edge tensors of its
edges in both directions. The face's new corners must already be in `env`, and the
neighbouring faces' corners must be those `transfers` were computed with.
"""
function set_face_edges!(
        env::CTMEnvironment, face, transfers, (right_basis, left_basis)
    )
    # Every corner of the face is `D = Λ^(1/m)`; the bases are rescaled by `Λ / D` and `1 / D`
    # so that each pass divides by one corner per step.
    roots = corner_diagonal(env, first(face))
    m = length(face)
    right_projectors =
        right_blocks(env, face, transfers, right_basis * Diagonal(roots .^ (m - 1)))
    left_projectors =
        left_blocks(env, face, transfers, Diagonal(inv.(roots)) * left_basis)
    for edge in face
        reversed = reverse(edge)
        previous = prevedge(env.embedding, reversed)
        env.edgetensors[reversed] =
            right_projectors[edge] * inverse(cornertensor(env, reversed))
        env.edgetensors[edge] = left_projectors[edge] * inverse(cornertensor(env, previous))
    end
    return env
end

"""
    face_update!(env, tn, face; maxdim, alg) -> env

Replace the corners, bonds and edge tensors of the `NamedFace` `face` with the MP-BP solution of the face given the rest of `env`. The new subspace basis is aligned
onto the one from the face's previous update, so the edge tensors converge entry by entry.
"""
function face_update!(
        env::CTMEnvironment,
        tn,
        face::NamedFace;
        maxdim::Integer,
        alg::AbstractAlgorithm
    )
    transfers = transfer_tensors(tn, env, face)
    corners, bases = face_solve(
        env.embedding, face, transfers,
        Dictionary(face, [bond(env, edge) for edge in face]);
        maxdim, alg, reference = get(env.bases, face, nothing)
    )
    set_face_corners!(env, face, corners)
    set_face_bases!(env, face, bases)
    set_face_edges!(env, face, transfers, bases)
    return env
end
