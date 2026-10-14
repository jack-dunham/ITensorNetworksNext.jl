module ITensorNetworksNextGradedArraysExt

using GradedArrays: AbstractGradedOneTo, FusedGradedMatrix, FusedGradedVector, dual,
    fusedgradedmatrix, isdual, sectordata, sectors
using ITensorBase: NamedUnitRange, unnamed
using ITensorNetworksNext:
    ITensorNetworksNext, KeepDominant, alignment, diagonal_tensor, left_invariant_rows
using LinearAlgebra: Diagonal
using MatrixAlgebraKit: MatrixAlgebraKit, findtruncated
using TensorAlgebra: unmatricize

# Each value is repeated by its sector's quantum dimension, so `maxdim` counts kept dimensions and
# the copies of one value, being equal, are never split; the kept values are returned per sector.
function MatrixAlgebraKit.findtruncated(values::FusedGradedVector, strategy::KeepDominant)
    blocks = collect(sectordata(values))
    dims = length.(sectors(only(axes(values))))
    owners =
        [(i, j) for i in eachindex(blocks) for j in eachindex(blocks[i]) for _ in 1:dims[i]]
    repeated = [blocks[i][j] for (i, j) in owners]
    kept = [Int[] for _ in blocks]
    for (i, j) in unique(owners[findtruncated(repeated, strategy)])
        push!(kept[i], j)
    end
    return kept
end

function ITensorNetworksNext.left_invariant_rows(
        matrix::FusedGradedMatrix, right_basis::FusedGradedMatrix
    )
    blocks = sectordata(matrix)
    rows = [
        sector => left_invariant_rows(blocks[sector], size(block, 2))
            for (sector, block) in pairs(sectordata(right_basis))
    ]
    return fusedgradedmatrix(rows, dual(axes(right_basis, 2)), axes(right_basis, 1))
end

function ITensorNetworksNext.alignment(
        overlaps::FusedGradedMatrix, eigenvalues::FusedGradedVector, degeneracy_rtol
    )
    values = sectordata(eigenvalues)
    parts = [
        sector => alignment(block, values[sector], degeneracy_rtol)
            for (sector, block) in pairs(sectordata(overlaps))
    ]
    codomain, domain = axes(overlaps, 1), dual(axes(overlaps, 2))
    transform = fusedgradedmatrix([s => part[1] for (s, part) in parts], codomain, domain)
    inverse = fusedgradedmatrix([s => part[2] for (s, part) in parts], domain, codomain)
    aligned = copy(eigenvalues)
    for (sector, part) in parts
        copyto!(sectordata(aligned)[sector], part[3])
    end
    return transform, inverse, aligned
end

# A diagonal map runs from its dual index to its non-dual one; the diagonal is unchanged by the swap.
function ITensorNetworksNext.diagonal_tensor(diagonal::FusedGradedVector, row, column)
    isdual(unnamed(row)) && return diagonal_tensor(diagonal, column, row)
    blocks = [
        sector => Matrix(Diagonal(block)) for
            (sector, block) in pairs(sectordata(diagonal))
    ]
    matrix = fusedgradedmatrix(blocks, unnamed(row), dual(unnamed(column)))
    return unmatricize(matrix, (row,), (dual(column),))
end
# A dense diagonal on graded bonds, such as the eigenvalues of a face not yet solved.
function ITensorNetworksNext.diagonal_tensor(
        diagonal::Vector, row::NamedUnitRange{<:Any, <:Any, <:AbstractGradedOneTo}, column
    )
    space = isdual(unnamed(row)) ? unnamed(column) : unnamed(row)
    return diagonal_tensor(FusedGradedVector(diagonal, space), row, column)
end

end
