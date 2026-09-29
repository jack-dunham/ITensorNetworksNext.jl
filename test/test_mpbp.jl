using Graphs: dst, edges, src, vertices
using ITensorBase: Index, NamedTensor, inds, name
using ITensorNetworksNext.ITensorNetworkGenerators: ising_network
using ITensorNetworksNext: DenseEig, ITensorNetwork, beliefpropagation, bethe_free_entropy,
    contract_network, ctm_environment, ctmrg, darts, expect, face_update!,
    hexagonal_position, invariant_subspace, leftface, linkinds, normnetwork,
    planar_embedding, tensornetwork, vertex_scalar, vertex_term
using LinearAlgebra: Diagonal, I, inv, norm
using NamedGraphs: all_edges, incident_edges, named_grid, named_hexagonal_lattice_graph
using StableRNGs: StableRNG
using Test: @test, @test_throws, @testset

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

    @testset "`invariant_subspace` keeps complex-conjugate pairs together" begin
        # Eigenvalues 3, 1 ± i, 0.5: moduli 3, √2, √2, 0.5.
        B = [3.0 0 0 0; 0 1 -1 0; 0 1 1 0; 0 0 0 0.5]
        S = [1.0 2 0 1; 0 1 3 0; 1 0 1 2; 0 1 0 1]
        Λ = S * B / S
        for (maxdim, χ) in ((1, 1), (2, 1), (3, 3), (4, 4))
            VR, VL, λ = invariant_subspace(DenseEig(), Λ, maxdim)
            @test length(λ) == χ
            @test Λ * VR ≈ VR * Diagonal(λ)
            @test VL * Λ ≈ Diagonal(λ) * VL
            @test VL * VR ≈ Matrix(I, χ, χ)
        end
        _, _, λ = invariant_subspace(DenseEig(), Diagonal([1.0, 5.0e-11, 0.0]), 3)
        @test length(λ) == 2
    end

    function run_sweeps!(env, tn; maxdim, nsweeps)
        for _ in 1:nsweeps, f in eachindex(env.embedding.faces)
            face_update!(env, tn, f; maxdim, alg = DenseEig())
        end
        return env
    end
    function bp_environment(tn, g, emb)
        cache = beliefpropagation(
            tn, bp_messages(tn, g);
            stopping_criterion = (; maxiter = 100, tol = 1.0e-14)
        )
        return ctm_environment(tn, emb, cache)
    end

    @testset "Face update reaches the exact Z ($lattice)" for (lattice, g, pos) in (
            ("2×2 grid, one face", named_grid((2, 2)), v -> v), LATTICES...,
        )
        tn, _ = ising_setup(g, 0.4)
        emb = planar_embedding(g, pos)
        env = run_sweeps!(bp_environment(tn, g, emb), tn; maxdim = 16, nsweeps = 30)
        z_exact = contract_network(tn)[]
        @test exp(bethe_free_entropy(tn, env)) ≈ z_exact rtol = 1.0e-10
    end

    @testset "Converged Z_B is stationary under perturbations" begin
        g = named_grid((4, 4))
        tn, _ = ising_setup(g, 0.4)
        emb = planar_embedding(g, v -> v)
        env = run_sweeps!(bp_environment(tn, g, emb), tn; maxdim = 2, nsweeps = 60)
        logz = bethe_free_entropy(tn, env)
        rng = StableRNG(1)
        d = first(
            filter(
                d -> all(!iszero, (leftface(emb, d), leftface(emb, reverse(d)))),
                darts(emb)
            )
        )
        # A single edge tensor leaves log Z_B unchanged at the fixed point, so the pair on
        # `d` and `reverse(d)` is perturbed together.
        for field in (:edgetensors, :corners)
            ts = [getfield(env, field)[x] for x in (d, reverse(d))]
            δts = [randn(rng, eltype(t), Tuple(inds(t))) for t in ts]
            change(ε) = begin
                env′ = copy(env)
                for (x, t, δt) in zip((d, reverse(d)), ts, δts)
                    getfield(env′, field)[x] = t + ε * norm(t) / norm(δt) * δt
                end
                abs(bethe_free_entropy(tn, env′) - logz)
            end
            @test change(1.0e-3) / change(1.0e-4) > 50
        end
    end

    sc = (; maxiter = 100, tol = 1.0e-12)

    @testset "χ = 1 fixed point is the BP fixed point ($lattice)" for (lattice, g, pos) in
        LATTICES

        tn, _ = ising_setup(g, 0.4)
        emb = planar_embedding(g, pos)
        cache = beliefpropagation(
            tn, bp_messages(tn, g);
            stopping_criterion = (; maxiter = 100, tol = 1.0e-14)
        )
        env = ctmrg(tn, emb; maxdim = 1, stopping_criterion = sc)
        @test exp(bethe_free_entropy(tn, env)) ≈ exp(bethe_free_entropy(tn, cache)) rtol =
            1.0e-8
    end

    @testset "Z_B and magnetisation converge to exact ($lattice)" for (lattice, g, pos) in
        LATTICES

        β = 0.4
        tn, l = ising_setup(g, β)
        emb = planar_embedding(g, pos)
        z = contract_network(tn)[]
        err(χ) = abs(
            exp(
                bethe_free_entropy(tn, ctmrg(tn, emb; maxdim = χ, stopping_criterion = sc))
            ) / z - 1
        )
        @test err(1) > err(2)
        env = ctmrg(tn, emb; maxdim = 8, stopping_criterion = sc)
        @test exp(bethe_free_entropy(tn, env)) ≈ z rtol = 1.0e-10
        v = first(vertices(g))
        tn_sz = ising_network(l, β, g; h = 0.1, sz_vertices = [v])
        @test expect(tn, env, v, tn_sz[v]) ≈ contract_network(tn_sz)[] / z rtol = 1.0e-10
    end

    @testset "Z_B is invariant under link gauge transformations" begin
        g = named_grid((4, 4))
        tn, _ = ising_setup(g, 0.4)
        emb = planar_embedding(g, v -> v)
        z = exp(bethe_free_entropy(tn, ctmrg(tn, emb; maxdim = 2, stopping_criterion = sc)))
        rng = StableRNG(2)
        tn_g = ITensorNetwork(Dict(v => tn[v] for v in vertices(tn)))
        for e in edges(g)
            i = only(linkinds(tn, e))
            j = Index(length(i))
            G = randn(rng, length(i), length(i)) + 3I
            tn_g[src(e)] = tn_g[src(e)] * NamedTensor(G, (name(i), name(j)))
            tn_g[dst(e)] = tn_g[dst(e)] * NamedTensor(inv(G), (name(j), name(i)))
        end
        z_g = exp(
            bethe_free_entropy(tn_g, ctmrg(tn_g, emb; maxdim = 2, stopping_criterion = sc))
        )
        @test z_g ≈ z rtol = 1.0e-10
    end

    @testset "`NormNetwork` of a random state" begin
        g = named_grid((3, 3))
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        s = Dict(v => Index(2) for v in vertices(g))
        rng = StableRNG(3)
        ψ = tensornetwork(vertices(g)) do v
            return randn(rng, (s[v], map(e -> l[e], incident_edges(g, v))...))
        end
        nn = normnetwork(ψ)
        emb = planar_embedding(g, v -> v)
        env = ctmrg(nn, emb; maxdim = 8, stopping_criterion = sc)
        @test exp(bethe_free_entropy(nn, env)) ≈ contract_network(nn)[] rtol = 1.0e-10
    end

    @testset "Complex element type" begin
        g = named_grid((3, 3))
        tn, _ = ising_setup(g, 0.4 + 0im)
        emb = planar_embedding(g, v -> v)
        env = ctmrg(tn, emb; maxdim = 8, stopping_criterion = sc)
        @test exp(bethe_free_entropy(tn, env)) ≈ contract_network(tn)[] rtol = 1.0e-10
    end

    @testset "`maxdim` above the available rank" begin
        g = named_hexagonal_lattice_graph(2, 2)
        tn, _ = ising_setup(g, 0.4)
        emb = planar_embedding(g, hexagonal_position)
        z_b = exp(
            bethe_free_entropy(tn, ctmrg(tn, emb; maxdim = 64, stopping_criterion = sc))
        )
        @test isfinite(z_b)
        @test z_b ≈ contract_network(tn)[] rtol = 1.0e-10
    end

    @testset "Unconverged run throws" begin
        g = named_grid((4, 4))
        tn, _ = ising_setup(g, 0.4)
        emb = planar_embedding(g, v -> v)
        @test_throws ErrorException ctmrg(
            tn, emb; maxdim = 4, stopping_criterion = (; maxiter = 1, tol = 1.0e-14)
        )
    end
end
