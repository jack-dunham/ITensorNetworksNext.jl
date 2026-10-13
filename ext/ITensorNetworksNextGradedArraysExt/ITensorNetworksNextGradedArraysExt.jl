module ITensorNetworksNextGradedArraysExt

using GradedArrays:
    FusedGradedMatrix, FusedGradedVector, dual, fusedgradedmatrix, sectordata, sectors
using ITensorNetworksNext: ITensorNetworksNext, KeepDominant, alignment, left_invariant_rows
using MatrixAlgebraKit: MatrixAlgebraKit, findtruncated

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

end
