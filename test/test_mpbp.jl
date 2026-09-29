using Graphs: edges, vertices
using ITensorBase: Index, inds
using ITensorNetworksNext.ITensorNetworkGenerators: ising_network
using ITensorNetworksNext: DenseEig, beliefpropagation, bethe_free_entropy,
    contract_network, corner, ctm_environment, darts, edgetensor, face_update!,
    hexagonal_position, invariant_subspace, leftface, linkinds, planar_embedding,
    vertex_scalar, vertex_term
using LinearAlgebra: Diagonal, I, norm
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
        d = first(
            filter(
                d -> all(!iszero, (leftface(emb, d), leftface(emb, reverse(d)))),
                darts(emb)
            )
        )
        # At a fixed point log Z_B is exactly constant in any single tensor, so the tensors on
        # `d` and `reverse(d)` are perturbed together to give a nonzero second-order change.
        for field in (:edgetensors, :corners)
            ts = [getfield(env, field)[x] for x in (d, reverse(d))]
            δts = [randn(eltype(t), Tuple(inds(t))) for t in ts]
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
end
