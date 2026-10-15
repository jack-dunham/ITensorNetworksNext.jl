using Adapt: adapt
using GradedArrays: U1, gradedrange
using Graphs: edges, vertices
using ITensorBase:
    ITensor, Index, inds, inputnames, name, operator, outputnames, state, unnamed
using ITensorNetworksNext: BlockedMessageUpdate, NormNetwork, QuadraticFormNetwork,
    SimpleMessageUpdate, beliefpropagation, branamemap, default_nblocks, insertlink!,
    message_environment, message_update!, tensornetwork, updated_message
using JLArrays: JLArray
using LinearAlgebra: norm
using NamedGraphs: NamedEdge, incident_edges, named_grid, named_path_graph
using StableRNGs: StableRNG
using TensorAlgebra: MatricizeContract, TensorOperationsContract
using TensorOperations: TensorOperations
using Test: @test, @test_throws, @testset

function random_network(rng, ::Type{T}, g; χ = 4) where {T}
    l = Dict(e => Index(χ) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    return tensornetwork(vertices(g)) do v
        return randn(rng, T, (Index(2), map(e -> l[e], incident_edges(g, v))...))
    end
end

# The quadratic form of a random state on `g` around a random operator acting on each site alone.
# With `link`, the operators on the first two vertices also share an index of dimension 3.
function random_quadratic_form(rng, ::Type{T}, g; χ = 4, d = 2, link = false) where {T}
    l = Dict(e => Index(χ) for e in edges(g))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
    s = Dict(v => Index(d) for v in vertices(g))
    ket = tensornetwork(vertices(g)) do v
        return randn(rng, T, (s[v], map(e -> l[e], incident_edges(g, v))...))
    end
    out = Dict(v => Index(d) for v in vertices(g))
    vs = collect(vertices(g))
    L = Index(3)
    ops = tensornetwork(vertices(g)) do v
        is = (link && v in vs[1:2]) ? (out[v], s[v], L) : (out[v], s[v])
        return randn(rng, T, is)
    end
    op = operator(ops, [name(out[v]) for v in vs], [name(s[v]) for v in vs])
    return QuadraticFormNetwork(ket, op)
end

# `adapt` on a `NamedTensorOperator` returns a bare `ITensor`, so the operator is rebuilt.
function adapt_message(to, m)
    return operator(adapt(to, state(m)), outputnames(m), inputnames(m))
end

function relative_difference(a, b)
    a_host = ITensor(Array(unnamed(state(a))), inds(state(a)))
    return norm(a_host - state(b)) / norm(state(b))
end

# Records the arrays it allocates as temporaries and the arrays it is asked to free.
struct RecordingAllocator
    temporaries::Vector{Any}
    freed::Vector{Any}
end
function TensorOperations.tensoralloc(ttype, structure, ::Val{true}, a::RecordingAllocator)
    C = TensorOperations.tensoralloc(ttype, structure, Val(true))
    push!(a.temporaries, C)
    return C
end
function TensorOperations.tensorfree!(C, a::RecordingAllocator)
    push!(a.freed, C)
    return nothing
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

    @testset "a real ket and complex messages" begin
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
        # The element-type check runs before any contraction, so no GPU is needed to reach it.
        cutensor = TensorOperationsContract(; backend = TensorOperations.cuTENSORBackend())
        @test_throws ArgumentError updated_message(
            BlockedMessageUpdate(; contract_alg = cutensor), cache, nn,
            NamedEdge((2, 2) => (2, 3))
        )
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

    @testset "every temporary allocation is freed" begin
        rng = StableRNG(1234)
        nn = NormNetwork(random_network(rng, ComplexF64, named_grid((3, 3))))
        cache = swept_cache(nn)
        allocator = RecordingAllocator([], [])
        algorithm = BlockedMessageUpdate(;
            nblocks = 3, contract_alg = TensorOperationsContract(; allocator)
        )
        edge = NamedEdge((2, 2) => (2, 3))
        simple = message_update!(SimpleMessageUpdate(), map(identity, cache), nn, edge)
        blocked = message_update!(algorithm, map(identity, cache), nn, edge)
        @test relative_difference(blocked[edge], simple[edge]) <= 1.0e-12
        # (2, 2) has three incoming messages, so each of the three blocks makes three.
        @test length(allocator.temporaries) >= 9
        # TensorOperations frees its own scratch copies through their `Memory`, so match by pointer.
        @test all(
            a -> any(f -> pointer(f) == pointer(a), allocator.freed),
            allocator.temporaries
        )
    end

    @testset "a contraction algorithm other than TensorOperations" begin
        rng = StableRNG(1234)
        nn = NormNetwork(random_network(rng, ComplexF64, named_grid((3, 3))))
        cache = swept_cache(nn)
        algorithm = BlockedMessageUpdate(; nblocks = 3, contract_alg = MatricizeContract())
        for edge in edges(cache)
            simple = message_update!(SimpleMessageUpdate(), map(identity, cache), nn, edge)
            blocked = message_update!(algorithm, map(identity, cache), nn, edge)
            @test relative_difference(blocked[edge], simple[edge]) <= 1.0e-12
        end
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

    @testset "quadratic form with a product operator, T=$T" for T in (Float64, ComplexF64)
        qf = random_quadratic_form(StableRNG(1234), T, named_grid((3, 3)))
        cache = swept_cache(qf)
        for edge in edges(cache)
            simple = message_update!(SimpleMessageUpdate(), map(identity, cache), qf, edge)
            blocked = message_update!(
                BlockedMessageUpdate(; nblocks = 3), map(identity, cache), qf, edge
            )
            @test relative_difference(blocked[edge], simple[edge]) <= 1.0e-12
        end

        # Only the operator at an edge's source vertex has to act on its site alone.
        linked = random_quadratic_form(StableRNG(1234), T, named_path_graph(3); link = true)
        linked_cache = message_environment(one, linked)
        @test_throws ArgumentError updated_message(
            BlockedMessageUpdate(), linked_cache, linked, NamedEdge(1 => 2)
        )
        @test updated_message(
            BlockedMessageUpdate(), linked_cache, linked, NamedEdge(3 => 2)
        ) isa ITensor
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
