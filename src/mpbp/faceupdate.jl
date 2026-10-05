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
# The cut at `face`'s last edge: the indices its last and first transfer tensors share, sorted
# by name so the stored bases' rows keep one order however the transfer tensors are laid out.
function face_cut(face, transfers)
    shared = commoninds(transfers[last(face)], transfers[first(face)])
    return Tuple(sort(collect(shared); by = name))
end

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
    face_solve(face, transfers; maxdim, alg, reference = nothing)
        -> eigenvalues, right_basis, left_basis

Solve the `NamedFace` `face` from its normalised corner transfer tensors `transfers` keyed
by edge. Reads no environment, so the transfer tensors can be computed by whichever ranks own
the vertices. `eigenvalues` are the kept eigenvalues of the product of the face's transfer
tensors, from which `set_face_corners!` builds the corners; `right_basis` and `left_basis`
span the invariant subspace, `face_projectors` builds the projectors from them, and a later
solve takes the pair as `reference`.
"""
function face_solve(
        face, transfers; maxdim::Integer, alg::AbstractAlgorithm, reference = nothing
    )
    product = CornerTransferProduct(collect(transfers), face_cut(face, transfers))
    right_basis, left_basis, eigenvalues =
        invariant_subspace(alg, product, maxdim; reference)
    # Ill-conditioned eigenvalues or bases make the edge tensors written from them inaccurate.
    @debug(
        "face_solve", face, bond_dim = length(eigenvalues),
        eigenvalue_condition = maximum(abs, eigenvalues) / minimum(abs, eigenvalues),
        projector_condition = opnorm(left_basis) * opnorm(right_basis),
    )
    return eigenvalues, right_basis, left_basis
end

# Writes each corner of `face`, keeping a bond's index while its dimension is unchanged. Bonds
# are read off corners, so all are read before any is written.
function set_face_corners!(env::CTMEnvironment, face, eigenvalues)
    # Every corner holds the same `m`-th root of the eigenvalues, which keeps each corner's
    # condition number the `m`-th root of the eigenvalues' instead of concentrating it in one.
    m = length(face)
    roots = if all(value -> isreal(value) && real(value) > 0, eigenvalues)
        real.(eigenvalues) .^ (1 / m)
    else
        complex.(eigenvalues) .^ (1 / m)
    end
    new_bonds = map(Dictionary(face, face)) do edge
        old_bond = bond(env, edge)
        return length(old_bond) == length(roots) ? old_bond : Index(length(roots))
    end
    for edge in face
        row, column = new_bonds[edge], new_bonds[nextedge(env.embedding, edge)]
        env.cornertensors[edge] = diagonal_tensor(roots, row, column)
    end
    return env
end

function set_face_bases!(env::CTMEnvironment, face, right_basis, left_basis)
    env.bases[face] = (right_basis, left_basis)
    return env
end

"""
    face_projectors(env, face, transfers, right_basis, left_basis)
        -> right_projectors, left_projectors

The right and left projectors at each edge of `face`, built from its bases by carrying them
around the face through `transfers`. The face's new corners must already be in `env`.
"""
function face_projectors(env::CTMEnvironment, face, transfers, right_basis, left_basis)
    # Every corner of the face is `D = Λ^(1/m)`; the bases are rescaled by `Λ / D` and `1 / D`
    # so that each pass divides by one corner per step.
    roots = corner_diagonal(env, first(face))
    m = length(face)
    right_projectors =
        right_blocks(env, face, transfers, right_basis * Diagonal(roots .^ (m - 1)))
    left_projectors =
        left_blocks(env, face, transfers, Diagonal(inv.(roots)) * left_basis)
    return right_projectors, left_projectors
end

# Writes both edge tensors on the face edge `edge` from that face's projectors at its cut. It
# reads the neighbouring face's corners, which must be those the projectors were built with.
function set_face_edge!(env::CTMEnvironment, edge, right_projector, left_projector)
    reversed = reverse(edge)
    previous = prevedge(env.embedding, reversed)
    env.edgetensors[reversed] = right_projector * inverse(cornertensor(env, reversed))
    env.edgetensors[edge] = left_projector * inverse(cornertensor(env, previous))
    return env
end

"""
    face_update!(env, tn, face; maxdim, alg) -> env

Replace the corners, bonds and edge tensors of the `NamedFace` `face` with the MP-BP solution
of the face given the rest of `env`. The new subspace basis is aligned onto the one from the
face's previous update, so the edge tensors converge entry by entry.
"""
function face_update!(
        env::CTMEnvironment,
        tn,
        face::NamedFace;
        maxdim::Integer,
        alg::AbstractAlgorithm
    )
    transfers = transfer_tensors(tn, env, face)

    eigenvalues, right_basis, left_basis = face_solve(
        face, transfers; maxdim, alg, reference = get(env.bases, face, nothing)
    )

    set_face_corners!(env, face, eigenvalues)
    set_face_bases!(env, face, right_basis, left_basis)

    right_projectors, left_projectors = face_projectors(
        env, face, transfers,
        right_basis, left_basis
    )

    for edge in face
        right_projector = right_projectors[edge]
        left_projector = left_projectors[edge]

        set_face_edge!(env, edge, right_projector, left_projector)
    end

    return env
end
