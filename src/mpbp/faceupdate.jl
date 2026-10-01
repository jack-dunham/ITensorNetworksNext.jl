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
# and divided by the face's eigenvalue corner; it has the bond of the face edge after it.
function left_blocks(tn, env::CTMEnvironment, face::Int, transfers, left_basis)
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
    for edge in face_edges[1:(end - 1)]
        block = blocks[prev_edge(embedding, edge)] * transfers[edge]
        block = if edge == divided_edge
            inverse_corner(env, divided_edge) * block
        else
            rename(block, bond(env, edge) => bond_after(edge))
        end
        set!(blocks, edge, block)
    end
    return blocks
end

# Inverse of `c[edge]` as a map between its two bonds, so that an edge tensor contracted
# with `c[edge]` and then with this is unchanged.
function inverse_corner(env::CTMEnvironment, edge)
    a, b = bond(env, edge), bond(env, next_edge(env.embedding, edge))
    return unmatricize(inv(matricize(corner(env, edge), (a,), (b,))), (b,), (a,))
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
    product = CornerTransferProduct(collect(transfers), cut_inds(tn, env, last(face_edges)))

    right_basis, left_basis, eigenvalues = invariant_subspace(alg, product, maxdim)
    bond_dim = length(eigenvalues)
    eigenvalue_corner = Matrix(Diagonal(eigenvalues))

    # The corners and projectors below take the face's new bonds from `env`.
    foreach(edge -> set!(env.bonds, edge, Index(bond_dim)), face_edges)
    elt = promote_type(eltype(right_basis), eltype(left_basis), eltype(eigenvalue_corner))
    for edge in face_edges
        bonds = ((bond(env, edge),), (bond(env, next_edge(embedding, edge)),))
        env.corners[edge] = if edge == eigenvalue_edge(face_edges)
            unmatricize(Matrix{elt}(eigenvalue_corner), bonds...)
        else
            id(elt, bonds...)
        end
    end

    right_projectors = right_blocks(tn, env, face, transfers, right_basis)
    left_projectors = left_blocks(tn, env, face, transfers, left_basis)
    for edge in face_edges
        reversed = reverse(edge)
        env.edgetensors[reversed] = right_projectors[edge] * inverse_corner(env, reversed)
        env.edgetensors[edge] =
            left_projectors[edge] * inverse_corner(env, prev_edge(embedding, reversed))
    end
    return env
end
