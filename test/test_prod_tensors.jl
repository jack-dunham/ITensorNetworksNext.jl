using Adapt: adapt
using Graphs: edges, src, vertices
using ITensorBase: ITensor, Index, NamedTensorOperator, inds, inputnames, name, operator,
    outputnames, setname, state, uniquename, unnamed
using ITensorNetworksNext: Greedy, ITensorNetwork, NormNetwork, Sliced, absorb_matrices!,
    beliefpropagation, conj_bratensor, incoming_messages, kettensor, linkinds,
    message_environment, prod_tensors, siteinds, tensornetwork
using JLArrays: JLArray
using NamedGraphs: NamedEdge, incident_edges, named_grid
using OMEinsumContractionOrders: ExhaustiveSearch, GreedyMethod, TreeSA
using TensorAlgebra: MatricizeContract, TensorOperationsContract
using TensorOperations: TensorOperations as TO
using Test: @test, @test_throws, @testset

# Records the arrays it allocates as temporaries and the arrays it is asked to free.
struct RecordingAllocator
    temporaries::Vector{Any}
    freed::Vector{Any}
end
function TO.tensoralloc(ttype, structure, ::Val{true}, a::RecordingAllocator)
    C = TO.tensoralloc(ttype, structure, Val(true))
    push!(a.temporaries, C)
    return C
end
function TO.tensorfree!(C, a::RecordingAllocator)
    push!(a.freed, C)
    return nothing
end

@testset "prod_tensors" begin
    @testset "Contract Vectors of ITensors" begin
        i, j, k = Index(2), Index(2), Index(5)
        A = [1.0 1.0; 0.5 1.0][i, j]
        B = [2.0, 1.0][i]
        C = [5.0, 1.0][j]
        D = [-2.0, 3.0, 4.0, 5.0, 1.0][k]

        ts = [A, B, C, D]
        ABCD_1 = prod_tensors(ts)
        ABCD_2 = prod_tensors(ts, Greedy())
        ABCD_3 = prod_tensors(ts, ExhaustiveSearch())
        ABCD_4 = prod_tensors(ts, GreedyMethod())
        ABCD_5 = prod_tensors(ts, TreeSA())
        @test ABCD_1 == ABCD_2 == ABCD_3
        @test ABCD_1 ≈ ABCD_4
        @test ABCD_1 ≈ ABCD_5
    end

    @testset "Contract One Dimensional Network" begin
        dims = (4, 4)
        g = named_grid(dims)
        l = Dict(e => Index(2) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        tn = tensornetwork(vertices(g)) do v
            is = map(e -> l[e], incident_edges(g, v))
            return randn(Tuple(is))
        end

        z1 = prod_tensors(tn)[]
        z2 = prod_tensors(tn, Greedy())[]
        z3 = prod_tensors(tn, ExhaustiveSearch())[]
        z4 = prod_tensors(tn, GreedyMethod())[]
        z5 = prod_tensors(tn, TreeSA())[]

        @test abs(z1 - z2) / abs(z1) <= 1.0e3 * eps(Float64)
        @test abs(z1 - z3) / abs(z1) <= 1.0e3 * eps(Float64)

        @test z1 ≈ z2
        @test z1 ≈ z3
        @test z1 ≈ z4
        @test z1 ≈ z5
    end

    @testset "Contract network with operators" begin
        i, j, k = Index(2), Index(2), Index(2)
        o = operator(randn(2, 2), (i,), (j,))     # output i, input j

        # A network mixing an operator with plain tensors previously threw a `convert`
        # `MethodError`; it now contracts, stays an operator, and matches the binary product.
        t = randn(2, 2)[j, k]
        r = prod_tensors([o, t])
        @test r isa NamedTensorOperator
        @test state(r) ≈ state(o * t)
        @test outputnames(r) == outputnames(o * t)
        @test inputnames(r) == inputnames(o * t)

        # An all-operator network is likewise preserved.
        o2 = operator(randn(2, 2), (j,), (k,))
        r2 = prod_tensors([o, o2])
        @test r2 isa NamedTensorOperator
        @test state(r2) ≈ state(o * o2)

        # A fully-contracted operator network reads out as a scalar via `[]`.
        f = randn(2, 2)[i, j]
        @test prod_tensors([o, f])[] ≈ (o * f)[]

        # An all-plain network is unaffected: it is not promoted to an operator.
        a = randn(2, 2)[i, j]
        b = randn(2, 2)[j, k]
        @test !(prod_tensors([a, b]) isa NamedTensorOperator)

        # Pairing is order-independent: a branching network with a surviving output/input pair
        # matches the binary product under any fold order (greedy contraction included).
        ip, mp, m, x = Index(2), Index(2), Index(2), Index(2)
        op = operator(randn(2, 2, 2, 2), (ip, mp), (i, m))
        u = randn(2, 2)[mp, x]
        w = randn(2, 2)[m, x]
        rb = prod_tensors([op, u, w])
        @test outputnames(rb) == outputnames((op * u) * w) == outputnames(op * (u * w))
        @test inputnames(rb) == inputnames((op * u) * w) == inputnames(op * (u * w))
    end

    @testset "`Sliced` matches `Greedy`, $arraytype" for arraytype in (Array, JLArray)
        g = named_grid((3, 3))
        l = Dict(e => Index(4) for e in edges(g))
        l = merge(l, Dict(reverse(e) => l[e] for e in edges(g)))
        network = tensornetwork(vertices(g)) do v
            return randn(ComplexF64, (Index(2), map(e -> l[e], incident_edges(g, v))...))
        end
        nn = NormNetwork(network)
        cache = beliefpropagation(
            nn, message_environment(one, nn); stopping_criterion = (; maxiter = 2)
        )
        # The message update at the centre towards (2, 3), whose open indices have length 4.
        edge = NamedEdge((2, 2) => (2, 3))
        v = src(edge)
        tensors =
            [kettensor(nn, v); incoming_messages(cache, edge)...; conj_bratensor(nn, v)]
        conjlist = [falses(length(tensors) - 1); true]
        reference = prod_tensors(tensors, Greedy(); conjlist)
        device_tensors = map(t -> adapt(arraytype, state(t)), tensors)
        for nblocks in (1, 2, 3, 4), contract_alg in (nothing, TensorOperationsContract())
            result = prod_tensors(device_tensors, Sliced(; nblocks, contract_alg); conjlist)
            @test unnamed(result) isa arraytype
            @test ITensor(Array(unnamed(result)), inds(result)) ≈ state(reference)
        end

        # Every intermediate is freed. TensorOperations frees its own scratch copies through
        # their `Memory`, so they are matched by pointer.
        allocator = RecordingAllocator([], [])
        contract_alg = TensorOperationsContract(; allocator)
        result = prod_tensors(device_tensors, Sliced(; nblocks = 2, contract_alg); conjlist)
        @test ITensor(Array(unnamed(result)), inds(result)) ≈ state(reference)
        @test !isempty(allocator.temporaries)
        @test all(
            a -> any(f -> pointer(f) == pointer(a), allocator.freed), allocator.temporaries
        )
    end

    @testset "absorb_matrices! matches * ($alg)" for alg in (
            MatricizeContract(), TensorOperationsContract(),
            TensorOperationsContract(; allocator = TO.BufferAllocator()),
        )
        legs = Index.((2, 3, 4))
        matrix(elt, i) = randn(elt, (setname(i, uniquename(name(i))), i))
        for n in 0:3
            ψ = randn(legs)
            matrices = [matrix(Float64, legs[k]) for k in 1:n]
            expected = foldl(*, matrices; init = copy(ψ))
            @test absorb_matrices!(alg, ψ, matrices...) ≈ expected
        end
        # TensorOperations forms the product in place, so it rejects a wider element type or a
        # non-square matrix.
        for (ψ, m) in (
                (randn(legs), matrix(ComplexF64, legs[1])),
                (randn(legs), randn((Index(5), legs[1]))),
            )
            if alg isa TensorOperationsContract
                @test_throws ArgumentError absorb_matrices!(alg, copy(ψ), m)
            else
                @test absorb_matrices!(alg, copy(ψ), m) ≈ ψ * m
            end
        end
    end
end
