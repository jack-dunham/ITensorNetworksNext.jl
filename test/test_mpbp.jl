using Graphs: edges, vertices
using ITensorBase: Index
using ITensorNetworksNext.ITensorNetworkGenerators: ising_network
using ITensorNetworksNext: beliefpropagation, bethe_free_entropy, contract_network,
    ctm_environment, hexagonal_position, linkinds, planar_embedding, vertex_scalar,
    vertex_term
using NamedGraphs: all_edges, named_grid, named_hexagonal_lattice_graph
using Test: @test, @testset

function ising_setup(g, β; h = 0.1, sz_vertices = [])
    ldict = Dict(e => Index(2) for e in edges(g))
    l(e) = get(() -> ldict[reverse(e)], ldict, e)
    return ising_network(l, β, g; h, sz_vertices), l
end

bp_messages(tn, g) = Dict(e => ones(Tuple(linkinds(tn, e))) for e in all_edges(g))

const LATTICES = (
    ("grid", named_grid((4, 4)), v -> v),
    ("hexagonal", named_hexagonal_lattice_graph(2, 2), hexagonal_position),
)

@testset "MP-BP" begin
    @testset "BP initialisation reproduces BP ($lattice)" for (lattice, g, pos) in LATTICES
        tn, _ = ising_setup(g, 0.4)
        emb = planar_embedding(g, pos)
        cache = beliefpropagation(
            tn, bp_messages(tn, g);
            stopping_criterion = (; maxiter = 100, tol = 1.0e-14)
        )
        env = ctm_environment(tn, emb, cache)
        @test bethe_free_entropy(tn, env) ≈ bethe_free_entropy(tn, cache) rtol = 1.0e-12
        for v in vertices(g)
            @test vertex_term(tn, env, v) ≈ vertex_scalar(tn, cache, v) rtol = 1.0e-12
        end
    end
end
