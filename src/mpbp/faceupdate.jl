using DiagonalArrays: diagview
using Dictionaries: Dictionary, set!
using Graphs: degree, dst, src
using ITensorBase: ITensorBase, Index, NamedOneTo, inds, name, names, state, unnamed
using LinearAlgebra: inv, norm

function cutinds(tn, env::CTMEnvironment, edge)
    return (linkinds(tn, edge)..., envinds(env, reverse(edge))...)
end

# Transfer matrix at `src(outgoing)` from the face edge before `outgoing` to `outgoing`.
function corner_transfer_matrix(tn, env::CTMEnvironment, outgoing)
    incoming = prevedge(env.embedding, outgoing)
    vertex = src(outgoing)
    exclude = (src(incoming), dst(outgoing))

    transfer = contract_network([[tn[vertex]]; environment_tensors(env, vertex; exclude)])

    # At a vertex of degree 2 nothing else carries the far face's bonds, so its corner supplies them.
    if degree(env.embedding.graph, vertex) == 2
        transfer = transfer * cornertensor(env, reverse(outgoing))
    end

    # A fixed leg order, the incoming cut then the outgoing one, lets `CornerTransferProduct` read
    # the cut off the legs in an order that does not depend on how the contraction laid them out.
    legs = name.((cutinds(tn, env, incoming)..., cutinds(tn, env, outgoing)...))
    @assert issetequal(names(transfer), legs)
    return ITensorBase.align(state(transfer), legs)
end

# Normalised transfer tensor at each vertex of `face`, keyed by the vertex.
function transfer_tensors(tn, env::CTMEnvironment, face::NamedFace)
    transfers = map(face) do edge
        # `state` drops the operator pairing a `NormNetwork` vertex tensor carries.
        transfer = state(corner_transfer_matrix(tn, env, edge))
        return transfer / norm(transfer)
    end
    return Dictionary(src.(face), transfers)
end

# Carries `block` from `start` along `edges`, each directed the way the walk travels, through the
# transfer tensor where it enters each edge and the inverse of the corner where it leaves.
function carried_blocks(start, edges, transfers, corners, block)
    blocks = Dictionary([start], [block])
    for edge in edges
        block = transfers[src(edge)] * block * diaginv(corners[dst(edge)])
        set!(blocks, edge, block)
    end
    return blocks
end

# Inverse of a diagonal two-index `tensor`, such as a corner, as a map between its indices.
function diaginv(tensor)
    row, column = inds(tensor)
    return diagonal_tensor(inv.(diagview(unnamed(tensor))), row, column)
end

"""
    face_solve(transfers; maxdim, alg, reference = nothing)
        -> eigenvalues, right_basis, left_basis

Solve a face from its normalised corner transfer tensors `transfers`, keyed by vertex in the
face's order. Reads no environment, so the transfer tensors can be computed by whichever ranks own
the vertices. `eigenvalues` are the kept eigenvalues of the product of the face's transfer
tensors, from which `set_face_corners!` builds the corners; `right_basis` and `left_basis`
span the invariant subspace, `face_projectors` builds the projectors from them, and a later
solve takes the pair as `reference`.
"""
function face_solve(
        transfers; maxdim::Integer, alg::AbstractAlgorithm, reference = nothing
    )
    product = CornerTransferProduct(collect(transfers))
    right_basis, left_basis, eigenvalues =
        invariant_subspace(alg, product, maxdim; reference)
    # Ill-conditioned eigenvalues or bases make the edge tensors written from them inaccurate.
    @debug(
        "face_solve", bond_dim = length(eigenvalues),
        eigenvalue_condition = maximum(abs, eigenvalues) / minimum(abs, eigenvalues),
        projector_condition = norm(left_basis) * norm(right_basis),
    )
    return eigenvalues, right_basis, left_basis
end

# Writes each corner of `face` and each of its bonds, keeping a bond's index while its dimension
# is unchanged and otherwise naming it from `bondnames`, keyed by face edge, when given.
function set_face_corners!(env::CTMEnvironment, face, eigenvalues; bondnames = nothing)
    # Every corner holds the same `m`-th root of the eigenvalues, which keeps each corner's
    # condition number the `m`-th root of the eigenvalues' instead of concentrating it in one.
    m = length(face)
    roots = if all(value -> isreal(value) && real(value) > 0, eigenvalues)
        real.(eigenvalues) .^ (1 / m)
    else
        complex.(eigenvalues) .^ (1 / m)
    end
    new_bonds = map(Dictionary(face, face)) do edge
        old_bond = only(envinds(env, edge))
        length(old_bond) == length(roots) && return old_bond
        n = length(roots)
        return isnothing(bondnames) ? Index(n) : NamedOneTo(n, bondnames[edge])
    end
    for edge in face
        row, column = new_bonds[edge], new_bonds[nextedge(env.embedding, edge)]
        env.cornertensors[edge] = diagonal_tensor(roots, row, column)
        env.envinds[edge] = (new_bonds[edge],)
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
around the face through `transfers`. The face's new corners must already be in `env`. Left
projectors are keyed by the face's edges and right projectors by those edges reversed. Each
projector's legs are the cut at its edge, in the order `corner_transfer_matrix` fixes, then its
bond.
"""
function face_projectors(env::CTMEnvironment, face, transfers, right_basis, left_basis)
    # Every corner of the face is `D = Λ^(1/m)`; the bases are rescaled by `Λ / D` and `1 / D`
    # so that each pass divides by one corner per step.
    roots = corner_diagonal(env, first(face))

    corners = Dictionary(dst.(face), [cornertensor(env, edge) for edge in face])

    start, others = last(face), face[1:(end - 1)]

    # The scaling also takes the bases' shared index `k` to the bond of the first step's edge.
    k = last(inds(right_basis))
    right_bond, left_bond =
        only(envinds(env, last(others))), only(envinds(env, first(others)))
    right_block = right_basis * diagonal_tensor(roots .^ (length(face) - 1), k, right_bond)
    left_block = left_basis * diagonal_tensor(inv.(roots), k, left_bond)

    right_projectors = carried_blocks(
        reverse(start), reverse.(reverse(others)), transfers, corners, right_block
    )
    left_projectors = carried_blocks(start, others, transfers, corners, left_block)

    return right_projectors, left_projectors
end

# Writes both edge tensors on the face edge `edge` from that face's projectors at its cut. It
# reads the neighbouring face's corners, which must be those the projectors were built with.
function set_face_edge!(env::CTMEnvironment, edge, right_projector, left_projector)
    set_right_edge!(env, edge, right_projector)
    set_left_edge!(env, edge, left_projector)
    return env
end

# The edge tensor on `reverse(edge)`, at `src(edge)`, from the right projector at `edge`.
function set_right_edge!(env::CTMEnvironment, edge, right_projector)
    reversed = reverse(edge)
    env.edgetensors[reversed] = right_projector * diaginv(cornertensor(env, reversed))
    return env
end

# The edge tensor on `edge`, at `dst(edge)`, from the left projector at `edge`.
function set_left_edge!(env::CTMEnvironment, edge, left_projector)
    previous = prevedge(env.embedding, reverse(edge))
    env.edgetensors[edge] = left_projector * diaginv(cornertensor(env, previous))
    return env
end

"""
    face_update!(env, tn, face; maxdim, alg, bondnames = nothing) -> env

Replace the corners, bonds and edge tensors of the `NamedFace` `face` with the MP-BP solution
of the face given the rest of `env`. The new subspace basis is aligned onto the one from the
face's previous update, so the edge tensors converge entry by entry. `bondnames`, keyed by face
edge, names the bonds whose dimension changes; by default their names are new.
"""
function face_update!(
        env::CTMEnvironment,
        tn,
        face::NamedFace;
        maxdim::Integer,
        alg::AbstractAlgorithm,
        bondnames = nothing
    )
    transfers = transfer_tensors(tn, env, face)

    eigenvalues, right_basis, left_basis = face_solve(
        transfers; maxdim, alg, reference = get(env.bases, face, nothing)
    )

    set_face_corners!(env, face, eigenvalues; bondnames)
    set_face_bases!(env, face, right_basis, left_basis)

    right_projectors, left_projectors = face_projectors(
        env, face, transfers,
        right_basis, left_basis
    )

    for edge in face
        set_face_edge!(env, edge, right_projectors[reverse(edge)], left_projectors[edge])
    end

    return env
end
