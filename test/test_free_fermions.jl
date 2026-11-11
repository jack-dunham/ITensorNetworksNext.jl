using GradedArrays: U1
using Graphs: dst, edges, is_tree, src, vertices
using ITensorBase: Index, apply, inds, operator, prime
using ITensorNetworksNext: Greedy, NormNetwork, apply_operator, apply_operators,
    beliefpropagation, bratensor, incoming_edge_data, insertlink!, kettensor,
    message_environment, prod_tensors, tensornetwork
using LinearAlgebra: norm
using NamedGraphs: named_comb_tree, named_cycle_graph
using TensorAlgebra: project, project_aux
using TensorKitSectors: FermionNumber
using Test: @test, @testset

# Free fermions on a tree with a branching vertex and on a loop, where the fermion signs change
# the observables, evolved in imaginary time bond by bond and compared with the exact Slater
# determinant. The circuit is not truncated, so the comparison is exact.

const cdag_matrix = Float64[0 0; 1 0]
const c_matrix = Float64[0 1; 0 0]

# `project_aux` derives an auxiliary leg carrying the charge of a charge-shifting operator, so
# `c†` conserves charge with that leg left dangling, and `c†ᵢ cⱼ` pairs a `c†` and a `c` over one
# shared such leg.
function cdag(s::Index)
    return operator(project_aux(cdag_matrix, (prime(s),), (s,)), [prime(s)], [s])
end
function cdag_c(si::Index, sj::Index)
    a = project_aux(cdag_matrix, (prime(si),), (si,))
    aux = last(inds(a))
    b = project(reshape(c_matrix, 2, 2, 1), (prime(sj),), (sj, aux))
    return operator(a, [prime(si)], [si]) * operator(b, [prime(sj)], [sj])
end
hopping(si::Index, sj::Index) = cdag_c(si, sj) + cdag_c(sj, si)
number(s::Index) = operator(project([0 0; 0 1], (prime(s),), (s,)), [prime(s)], [s])

expectation(o, ψ) = (conj(ψ) * apply(o, ψ))[] / (conj(ψ) * ψ)[]

# `⟨c†ᵢ cⱼ⟩` of the full state `ψ` for every pair of vertices of `g`, with the densities on the
# diagonal.
function full_state_correlators(g, sites, ψ)
    vs = collect(vertices(g))
    return [
        expectation(i == j ? number(sites[i]) : cdag_c(sites[i], sites[j]), ψ) for
            i in vs, j in vs
    ]
end

# Belief-propagation estimate of `⟨ψ|op|ψ⟩ / ⟨ψ|ψ⟩` from the doubled `region` of `nn` and the
# messages coming into it.
function bp_expectation(op, nn, cache, region)
    messages = collect(incoming_edge_data(cache, region))
    ket = prod(v -> kettensor(nn, v), region)
    bras = [bratensor(nn, v) for v in region]
    numerator = prod_tensors([messages; apply(op, ket); bras], Greedy())[]
    return numerator / prod_tensors([messages; ket; bras], Greedy())[]
end

# `⟨c†ᵢ cⱼ⟩` over the vertices of `g` for fermions created on `occupied` and evolved by `exp(-τ h)`
# on the bonds `(v1, v2, τ)`: the state stays a Slater determinant with orbitals `Φ`, each gate
# multiplies `Φ` by `exp(-τ h)` on its two sites, and `⟨cᵢ† cⱼ⟩ = (Φ (Φᵀ Φ)⁻¹ Φᵀ)ᵢⱼ`.
function free_fermion_correlators(g, occupied, bonds)
    index = Dict(v => i for (i, v) in enumerate(vertices(g)))
    Φ = zeros(length(index), length(occupied))
    for (k, v) in enumerate(occupied)
        Φ[index[v], k] = 1
    end
    for (v1, v2, τ) in bonds
        ij = [index[v1], index[v2]]
        Φ[ij, :] = exp(-τ * [0 1; 1 0]) * Φ[ij, :]
    end
    return Φ * ((Φ' * Φ) \ Φ')
end

# The empty product state on `sites`, each vertex tensor scaled by `scale`, with its identity
# messages.
function empty_state(g, sites; scale = 1)
    ψ = tensornetwork(v -> scale * ones(sites[v]), vertices(g))
    for e in edges(g)
        insertlink!(ψ, e)
    end
    return ψ, message_environment(one, NormNetwork(ψ))
end

# Create a fermion on each site in `occupied`, then apply `exp(-τ h)` on the bonds `(v1, v2, τ)`
# by belief-propagation simple update. Returns the network and its site indices.
function bp_evolve(sector, g, occupied, bonds)
    sites = Dict(v => Index([sector(0) => 1, sector(1) => 1]) for v in vertices(g))
    ψ, env = empty_state(g, sites)
    for v in occupied
        ψ, env = apply_operator(cdag(sites[v]), ψ, env)
    end
    gates = [exp(-τ * hopping(sites[v1], sites[v2])) for (v1, v2, τ) in bonds]
    ψ, env = apply_operators(gates, ψ, env)
    return ψ, sites
end

# An even number of fermions on the cycle, where an odd number would be indistinguishable from
# hardcore bosons.
@testset "free fermions ($label)" for (label, g, occupied) in (
        ("comb tree", named_comb_tree((3, 2)), [(2, 1), (2, 2)]),
        ("cycle", named_cycle_graph(6), [1, 2, 4, 5]),
    )
    bonds = [(src(e), dst(e), 0.5) for _ in 1:3 for e in edges(g)]
    C = free_fermion_correlators(g, occupied, bonds)

    ψ, sites = bp_evolve(FermionNumber, g, occupied, bonds)
    # Every pair, not only neighbours, so the signs along the path between two sites count.
    @test full_state_correlators(g, sites, prod(ψ)) ≈ C atol = 1.0e-8

    # On a tree belief propagation is exact, so its densities and its `⟨c†ᵢ cⱼ⟩` across each
    # edge, in both directions, match the Slater determinant.
    if is_tree(g)
        nn = NormNetwork(ψ)
        stopping_criterion = (; maxiter = 50, tol = 1.0e-12)
        cache = beliefpropagation(nn, message_environment(one, nn); stopping_criterion)
        index = Dict(v => i for (i, v) in enumerate(vertices(g)))
        for v in vertices(g)
            bp = bp_expectation(number(sites[v]), nn, cache, [v])
            @test bp ≈ C[index[v], index[v]] atol = 1.0e-8
        end
        for e in edges(g), (i, j) in ((src(e), dst(e)), (dst(e), src(e)))
            bp = bp_expectation(cdag_c(sites[i], sites[j]), nn, cache, [src(e), dst(e)])
            @test bp ≈ C[index[i], index[j]] atol = 1.0e-8
        end
    end

    # Hardcore bosons go through the same circuit without the fermion signs and come out
    # different on these graphs, so the checks above are sensitive to the signs.
    ψ_boson, sites_boson = bp_evolve(U1, g, occupied, bonds)
    @test maximum(abs, full_state_correlators(g, sites_boson, prod(ψ_boson)) - C) > 0.05
end

# With `normalize`, a one-site gate divides its vertex by its norm against the incoming messages,
# and a two-site gate normalizes the singular values on its bond.
@testset "free fermions, normalize" begin
    g = named_comb_tree((3, 2))
    sites = Dict(
        v => Index([FermionNumber(0) => 1, FermionNumber(1) => 1]) for v in vertices(g)
    )
    v1, v2 = (2, 1), (2, 2)

    # The messages of the product state are the identity, so `c†` leaves its vertex at unit norm
    # and the vertices it does not act on keep theirs.
    ψ, env = empty_state(g, sites; scale = 3)
    ψ, env = apply_operator(cdag(sites[v1]), ψ, env; normalize = true)
    @test norm(ψ[v1]) ≈ 1
    @test all(norm(ψ[v]) ≈ 3 for v in vertices(g) if v ≠ v1)

    # Every vertex of the product state has unit norm, so its messages are exact and the gated
    # state has the norm of its normalized singular values, where `exp(-τ h)` alone would not.
    ψ, env = empty_state(g, sites)
    ψ, env = apply_operator(cdag(sites[v1]), ψ, env)
    gate = exp(-0.5 * hopping(sites[v1], sites[v2]))
    unnormalized = prod(first(apply_operator(gate, ψ, env)))
    normalized = prod(first(apply_operator(gate, ψ, env; normalize = true)))
    @test norm(unnormalized) > 1.1
    @test norm(normalized) ≈ 1
    @test normalized ≈ unnormalized / norm(unnormalized)
end
