using Adapt: adapt
using GradedArrays: U1, gradedrange
using Graphs: edges, vertices
using ITensorBase: ITensor, Index, inds, inputnames, operator, outputnames, state, unnamed
using ITensorNetworksNext: BlockedMessageUpdate, NormNetwork, SimpleMessageUpdate,
    beliefpropagation, branamemap, default_nblocks, insertlink!, message_environment,
    message_update!, tensornetwork, updated_message
using JLArrays: JLArray
using LinearAlgebra: norm
using NamedGraphs: NamedEdge, incident_edges, named_grid, named_path_graph
using StableRNGs: StableRNG
using TensorOperations: TensorOperations
using Test: @test, @test_throws, @testset

function random_network(rng, ::Type{T}, g; χ = 4) where {T}
    l = Dict(e => Index(χ) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    return tensornetwork(vertices(g)) do v
        return randn(rng, T, (Index(2), map(e -> l[e], incident_edges(g, v))...))
    end
end

# `adapt` on a `NamedTensorOperator` returns a bare `ITensor`, so the operator is rebuilt.
function adapt_message(to, m)
    return operator(adapt(to, state(m)), outputnames(m), inputnames(m))
end

function relative_difference(a, b)
    a_host = ITensor(Array(unnamed(state(a))), inds(state(a)))
    return norm(a_host - state(b)) / norm(state(b))
end

# Two sweeps from identity messages give messages that are not the identity.
function swept_cache(nn)
    return beliefpropagation(
        nn, message_environment(one, nn); stopping_criterion = (; maxiter = 2)
    )
end

@testset "BlockedMessageUpdate" begin
    @testset "matches SimpleMessageUpdate, T=$T, $arraytype, nblocks=$nblocks" for T in
            (
                Float64,
                ComplexF64,
            ),
            arraytype in (Array, JLArray),
            nblocks in (2, 3)

        rng = StableRNG(1234)
        network = random_network(rng, T, named_grid((3, 3)))
        nn = NormNetwork(network)
        cache = swept_cache(nn)
        # `NormNetwork(ket)` draws fresh bra names, so the device copy reuses the host map.
        device_nn = NormNetwork(
            tensornetwork(v -> adapt(arraytype, network[v]), vertices(network)),
            branamemap(nn)
        )
        device_cache = map(m -> adapt_message(arraytype, m), cache)
        algorithm = BlockedMessageUpdate(; nblocks)
        for edge in edges(cache)
            simple = message_update!(SimpleMessageUpdate(), map(identity, cache), nn, edge)
            blocked =
                message_update!(algorithm, map(identity, device_cache), device_nn, edge)
            @test unnamed(state(blocked[edge])) isa arraytype
            @test relative_difference(blocked[edge], simple[edge]) <= 1.0e-12
        end
    end

    @testset "promotes a real ket and complex messages" begin
        rng = StableRNG(1234)
        nn = NormNetwork(random_network(rng, Float64, named_grid((3, 3))))
        cache = map(swept_cache(nn)) do m
            s = state(m)
            return operator(
                ITensor(complex.(unnamed(s)), inds(s)),
                outputnames(m),
                inputnames(m)
            )
        end
        for edge in edges(cache)
            simple = message_update!(SimpleMessageUpdate(), map(identity, cache), nn, edge)
            blocked = message_update!(
                BlockedMessageUpdate(; nblocks = 3), map(identity, cache), nn, edge
            )
            @test eltype(unnamed(state(blocked[edge]))) === ComplexF64
            @test relative_difference(blocked[edge], simple[edge]) <= 1.0e-12
        end
    end

    @testset "leaf vertices, which have no incoming messages" begin
        rng = StableRNG(1234)
        nn = NormNetwork(random_network(rng, ComplexF64, named_path_graph(4)))
        cache = swept_cache(nn)
        for edge in (NamedEdge(1 => 2), NamedEdge(4 => 3))
            simple = message_update!(SimpleMessageUpdate(), map(identity, cache), nn, edge)
            blocked = message_update!(
                BlockedMessageUpdate(; nblocks = 10), map(identity, cache), nn, edge
            )
            @test relative_difference(blocked[edge], simple[edge]) <= 1.0e-12
        end
    end

    @testset "normalize = false" begin
        rng = StableRNG(1234)
        nn = NormNetwork(random_network(rng, ComplexF64, named_grid((3, 3))))
        cache = swept_cache(nn)
        edge = NamedEdge((2, 2) => (2, 3))
        simple = message_update!(
            SimpleMessageUpdate(; normalize = false), map(identity, cache), nn, edge
        )
        blocked = message_update!(
            BlockedMessageUpdate(; normalize = false, nblocks = 3),
            map(identity, cache), nn, edge
        )
        @test relative_difference(blocked[edge], simple[edge]) <= 1.0e-12
    end

    @testset "selected through `beliefpropagation`" begin
        rng = StableRNG(1234)
        nn = NormNetwork(random_network(rng, ComplexF64, named_grid((3, 3))))
        simple = beliefpropagation(
            nn, message_environment(one, nn); stopping_criterion = (; maxiter = 3)
        )
        blocked = beliefpropagation(
            nn, message_environment(one, nn);
            stopping_criterion = (; maxiter = 3),
            message_update_algorithm = BlockedMessageUpdate(; nblocks = 3)
        )
        @test all(e -> relative_difference(blocked[e], simple[e]) <= 1.0e-10, edges(simple))
    end

    @testset "automatic `nblocks`" begin
        @test isnothing(BlockedMessageUpdate().nblocks)
        cutensor = TensorOperations.cuTENSORBackend()
        MiB = 2^20
        @test default_nblocks(cutensor, 1 * MiB, 32) == 1
        @test default_nblocks(cutensor, 40 * MiB, 256) == 10
        @test default_nblocks(cutensor, 1024 * MiB, 256) == 16
        @test default_nblocks(cutensor, 40 * MiB, 4096) == 64
        @test default_nblocks(TensorOperations.StridedBLAS(), 1024 * MiB, 4096) == 1
        @test default_nblocks(BlockedMessageUpdate(), randn(4, 4, 4), 4) == 1
        @test default_nblocks(BlockedMessageUpdate(; nblocks = 3), randn(4, 4, 4), 4) == 3

        rng = StableRNG(1234)
        nn = NormNetwork(random_network(rng, ComplexF64, named_grid((3, 3))))
        cache = swept_cache(nn)
        edge = NamedEdge((2, 2) => (2, 3))
        simple = message_update!(SimpleMessageUpdate(), map(identity, cache), nn, edge)
        blocked = message_update!(BlockedMessageUpdate(), map(identity, cache), nn, edge)
        @test relative_difference(blocked[edge], simple[edge]) <= 1.0e-12
    end

    @testset "argument errors" begin
        @test_throws ArgumentError BlockedMessageUpdate(; workspace_limit = 2^20)
        @test_throws ArgumentError BlockedMessageUpdate(; nblocks = 0)
        @test_throws ArgumentError BlockedMessageUpdate(; nblocks = 2.5)

        rng = StableRNG(1234)
        site_range = gradedrange([U1(0) => 1, U1(1) => 1])
        path = named_path_graph(3)
        graded = tensornetwork(v -> randn(rng, (Index(site_range),)), vertices(path))
        for e in edges(path)
            insertlink!(graded, e)
        end
        graded_nn = NormNetwork(graded)
        @test_throws ArgumentError updated_message(
            BlockedMessageUpdate(), message_environment(one, graded_nn), graded_nn,
            NamedEdge(1 => 2)
        )
    end
end
