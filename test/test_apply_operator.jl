using AlgorithmsInterface: AlgorithmsInterface as AI
using GradedArrays: U1, gradedrange
using Graphs: dst, edges, src, vertices
using ITensorBase: Index, apply, inputnames, name, names, nametype, operator, outputnames,
    setname, uniquename
using ITensorNetworksNext: ITensorNetworksNext, BPApplyGate,
    BeliefPropagationEnvironmentPreparation, BufferedBPGateUpdate, MessageUpdateAlgorithm,
    NormNetwork, SimpleBPGateUpdate, SimpleMessageUpdate, StopWhenVertexRevisited,
    apply_operator, apply_operator!, apply_operators, apply_operators!, beliefpropagation,
    bp_gate_factorize!, bp_gate_restore!, bp_gate_split, branamemap, insertlink!,
    message_environment, message_gauge, tensornetwork
using LinearAlgebra: norm
using MatrixAlgebraKit: svd_trunc, truncrank
using NamedGraphs: named_cycle_graph, named_grid, named_path_graph
using Random: AbstractRNG
using StableRNGs: StableRNG
using TensorAlgebra: TensorOperationsContract
using TensorKitSectors: FermionParity
using TensorOperations: TensorOperations as TO
using Test: @test, @test_throws, @testset

const spinone = Base.OneTo(3)
const spinone_u1 = gradedrange([U1(2) => 1, U1(0) => 1, U1(-2) => 1])
const fermion = gradedrange([FermionParity(0) => 2, FermionParity(1) => 2])

function randn_operator(rng::AbstractRNG, elt::Type, domain_namedaxes)
    codomain_namedaxes = setname.(domain_namedaxes, uniquename.(name.(domain_namedaxes)))
    dual_domain_namedaxes = setname.(conj.(domain_namedaxes), name.(domain_namedaxes))
    data = randn(rng, elt, (codomain_namedaxes..., dual_domain_namedaxes...))
    return operator(data, name.(codomain_namedaxes), name.(domain_namedaxes))
end

# Build a random state by applying random gates layer by layer, carrying the belief
# propagation environment through the applications. The returned `env` is the environment
# the gate applications produced, ready to gauge the next application (belief-propagation
# convergence itself is covered separately in `test_beliefpropagation.jl`).
function random_state(rng::AbstractRNG, elt::Type, g, site_axes; nlayers, trunc)
    network = tensornetwork(vertices(g)) do v
        return randn(rng, elt, (site_axes[v],))
    end

    for edge in edges(g)
        insertlink!(network, edge)
    end

    env = message_environment(one, NormNetwork(network))
    for _ in 1:nlayers, e in edges(g)
        gate = randn_operator(rng, elt, (site_axes[src(e)], site_axes[dst(e)]))
        network, env = apply_operator(gate, network, env; trunc)
    end
    return network, env
end

# Counts the single-edge updates belief propagation performs.
struct CountingMessageUpdate <: MessageUpdateAlgorithm
    count::Base.RefValue{Int}
end
function ITensorNetworksNext.message_update!(
        algorithm::CountingMessageUpdate, cache, factors, edge
    )
    algorithm.count[] += 1
    return ITensorNetworksNext.message_update!(SimpleMessageUpdate(), cache, factors, edge)
end

@testset "apply_operator (T=$T, $label)" for (label, site_range) in (
            "spinone" => spinone, "spinone_u1" => spinone_u1, "fermion" => fermion,
        ),
        T in (Float32, Float64, ComplexF64)

    N = 4

    @testset "untruncated gates are exact (gauge-invariant)" begin
        rng = StableRNG(123)
        g = named_cycle_graph(N)
        site_axes = Dict(v => Index(site_range) for v in vertices(g))
        network, env = random_state(rng, T, g, site_axes; nlayers = 2, trunc = truncrank(4))

        for gate in (
                randn_operator(rng, T, (site_axes[2],)),
                randn_operator(rng, T, (site_axes[2], site_axes[3])),
            )
            gated, _ = apply_operator(gate, network, env)
            @test prod(gated) ≈ apply(gate, prod(network)) rtol = eps(real(T))^(1 / 3)
        end
    end

    @testset "truncated 2-site gate matches global optimal SVD (rank $k)" for k in 1:3
        rng = StableRNG(123)
        g = named_path_graph(N)
        site_axes = Dict(v => Index(site_range) for v in vertices(g))
        network, env = random_state(rng, T, g, site_axes; nlayers = 2, trunc = truncrank(4))

        gate = randn_operator(rng, T, (site_axes[2], site_axes[3]))
        gated_full = apply(gate, prod(network))
        left = [name(site_axes[v]) for v in 1:2]
        U, S, Vt = svd_trunc(gated_full, left; trunc = truncrank(k))
        gated, _ = apply_operator(gate, network, env; trunc = truncrank(k))

        @test prod(gated) ≈ U * S * Vt rtol = eps(real(T))^(1 / 3)
    end

    @testset "apply_operators applies a sequence" begin
        rng = StableRNG(123)
        g = named_cycle_graph(N)
        site_axes = Dict(v => Index(site_range) for v in vertices(g))
        network, env = random_state(rng, T, g, site_axes; nlayers = 2, trunc = truncrank(4))

        g1 = randn_operator(rng, T, (site_axes[2], site_axes[3]))
        g2 = randn_operator(rng, T, (site_axes[3], site_axes[4]))
        gated, _ = apply_operators([g1, g2], network, env)
        @test prod(gated) ≈ apply(g2, apply(g1, prod(network))) rtol =
            eps(real(T))^(1 / 3)
    end

    @testset "apply_operators with explicit vertices" begin
        rng = StableRNG(123)
        g = named_cycle_graph(N)
        site_axes = Dict(v => Index(site_range) for v in vertices(g))
        network, env = random_state(rng, T, g, site_axes; nlayers = 2, trunc = truncrank(4))
        rtol = eps(real(T))^(1 / 3)

        gates = [
            randn_operator(rng, T, (site_axes[2], site_axes[3])),
            randn_operator(rng, T, (site_axes[3],)),
            randn_operator(rng, T, (site_axes[3], site_axes[4])),
        ]
        gated, gated_env = apply_operators(gates, network, env)
        explicit, explicit_env =
            apply_operators(gates, network, env; vertices = [[2, 3], [3], [3, 4]])
        # The two runs mint different bond names, so compare name-independent quantities.
        @test prod(explicit) ≈ prod(gated) rtol = rtol
        for edge in edges(gated_env)
            @test norm(explicit_env[edge]) ≈ norm(gated_env[edge]) rtol = rtol
        end

        @test_throws ArgumentError apply_operators(gates, network, env; vertices = [[2, 3]])
        axis = Index(site_range)
        offnetwork = randn_operator(rng, T, (axis,))
        @test_throws ArgumentError(
            "operator input `$(name(axis))` is not on any tensor of the network"
        ) apply_operators([gates[1], offnetwork], network, env)
    end

    @testset "bp_gate_split names the new bond as requested" begin
        rng = StableRNG(123)
        g = named_path_graph(N)
        site_axes = Dict(v => Index(site_range) for v in vertices(g))
        network, env = random_state(rng, T, g, site_axes; nlayers = 2, trunc = truncrank(4))

        gate = randn_operator(rng, T, (site_axes[2], site_axes[3]))
        gated = copy(network)
        gated[2], gated[3] = copy(network[2]), copy(network[3])
        simple = SimpleBPGateUpdate()
        Q_2, R_2, inverse_roots_2 = bp_gate_factorize!(simple, gate, gated, env, 2, 3)
        Q_3, R_3, inverse_roots_3 = bp_gate_factorize!(simple, gate, gated, env, 3, 2)
        bondnames = (uniquename(nametype(network)), uniquename(nametype(network)))
        R_2, R_3, message_23, message_32 = bp_gate_split(
            simple, gate, R_2, R_3; trunc = nothing, normalize = false, bondnames
        )
        @test intersect(names(R_2), names(R_3)) == [bondnames[1]]
        for message in (message_23, message_32)
            @test only(inputnames(message)) == bondnames[1]
            @test only(outputnames(message)) == bondnames[2]
        end
        gated[2] = bp_gate_restore!(simple, Q_2, R_2, inverse_roots_2)
        gated[3] = bp_gate_restore!(simple, Q_3, R_3, inverse_roots_3)
        @test prod(gated) ≈ apply(gate, prod(network)) rtol = eps(real(T))^(1 / 3)
    end

    @testset "message roots gauge a vertex and undo it" begin
        rng = StableRNG(123)
        g = named_path_graph(N)
        site_axes = Dict(v => Index(site_range) for v in vertices(g))
        network, env = random_state(rng, T, g, site_axes; nlayers = 2, trunc = truncrank(4))
        rtol = eps(real(T))^(1 / 3)

        for (edge, v) in ((2 => 3, 3), (3 => 2, 2))
            x, y = message_gauge(env[edge])
            @test y * (x * network[v]) ≈ network[v] rtol = rtol
        end
    end
end

# The middle vertices have degree 3, so `BufferedBPGateUpdate` applies two message roots and
# overwrites `state[v]`.
@testset "apply_operators! and input preservation (T=$T)" for T in
    (Float32, Float64, ComplexF64)
    rng = StableRNG(123)
    g = named_grid((2, 3))
    site_axes = Dict(v => Index(spinone) for v in vertices(g))
    network, env = random_state(rng, T, g, site_axes; nlayers = 2, trunc = truncrank(4))
    gates = [
        randn_operator(rng, T, (site_axes[(1, 2)], site_axes[(2, 2)])),
        randn_operator(rng, T, (site_axes[(1, 2)],)),
        randn_operator(rng, T, (site_axes[(2, 2)], site_axes[(1, 2)])),
        randn_operator(rng, T, (site_axes[(1, 1)], site_axes[(1, 2)])),
    ]
    snapshot = Dict(v => copy(network[v]) for v in vertices(g))
    rtol = eps(real(T))^(1 / 3)
    buffered = BPApplyGate(; trunc = truncrank(2), subalgorithm = BufferedBPGateUpdate())

    @testset "apply_operators leaves its input unchanged" begin
        apply_operators(gates, network, env; trunc = truncrank(2))
        @test all(v -> network[v] == snapshot[v], vertices(g))
        apply_operator(gates[1], network, env; trunc = truncrank(2))
        @test all(v -> network[v] == snapshot[v], vertices(g))
        apply_operators(gates, network, env; operator_alg = buffered)
        @test all(v -> network[v] == snapshot[v], vertices(g))
        apply_operator(gates[1], network, env; alg = buffered)
        @test all(v -> network[v] == snapshot[v], vertices(g))
    end

    @testset "BufferedBPGateUpdate matches SimpleBPGateUpdate" begin
        gated, gated_env = apply_operators(gates, network, env; trunc = truncrank(2))
        buffered_state, buffered_env =
            apply_operators(gates, network, env; operator_alg = buffered)
        # The two runs mint different bond names, so compare name-independent quantities.
        @test prod(buffered_state) ≈ prod(gated) rtol = rtol
        for edge in edges(gated_env)
            @test norm(buffered_env[edge]) ≈ norm(gated_env[edge]) rtol = rtol
        end
    end

    @testset "BufferedBPGateUpdate with a BufferAllocator matches SimpleBPGateUpdate" begin
        contract_alg = TensorOperationsContract(; allocator = TO.BufferAllocator())
        subalgorithm = BufferedBPGateUpdate(; contract_alg)
        operator_alg = BPApplyGate(; trunc = truncrank(2), subalgorithm)
        gated, gated_env = apply_operators(gates, network, env; trunc = truncrank(2))
        buffered_state, buffered_env = apply_operators(gates, network, env; operator_alg)
        @test prod(buffered_state) ≈ prod(gated) rtol = rtol
        for edge in edges(gated_env)
            @test norm(buffered_env[edge]) ≈ norm(gated_env[edge]) rtol = rtol
        end
    end

    T <: Real && @testset "BufferedBPGateUpdate rejects complex messages" begin
        complex_env = copy(env)
        for edge in edges(env)
            complex_env[edge] = (1 + 0im) * env[edge]
        end
        buffered = BPApplyGate(; subalgorithm = BufferedBPGateUpdate())
        @test_throws ArgumentError apply_operator(
            gates[1], network, complex_env; alg = buffered
        )
    end

    @testset "apply_operators! matches apply_operators" begin
        gated, gated_env = apply_operators(gates, network, env; trunc = truncrank(2))
        inplace, inplace_env = apply_operators!(gates, network, env; trunc = truncrank(2))
        # The two runs mint different bond names, so compare name-independent quantities.
        @test prod(inplace) ≈ prod(gated) rtol = rtol
        for edge in edges(gated_env)
            @test norm(inplace_env[edge]) ≈ norm(gated_env[edge]) rtol = rtol
        end
    end

    @testset "apply_operator! names the new bond as requested ($subalgorithm)" for
        subalgorithm in (SimpleBPGateUpdate(), BufferedBPGateUpdate())
        algorithm = BPApplyGate(; subalgorithm)
        gated, gated_env = copy(network), copy(env)
        for v in vertices(g)
            gated[v] = copy(network[v])
        end
        bondnames = (uniquename(nametype(network)), uniquename(nametype(network)))
        v1, v2 = (1, 2), (2, 2)
        apply_operator!(
            algorithm, gated, gates[1], gated, gated_env; vertices = [v1, v2], bondnames
        )
        @test intersect(names(gated[v1]), names(gated[v2])) == [bondnames[1]]
        for message in (gated_env[v1 => v2], gated_env[v2 => v1])
            @test only(inputnames(message)) == bondnames[1]
            @test only(outputnames(message)) == bondnames[2]
        end
        @test prod(gated) ≈ apply(gates[1], prod(network)) rtol = rtol
    end
end

@testset "BufferedBPGateUpdate rejects graded storage" begin
    rng = StableRNG(123)
    g = named_path_graph(3)
    site_axes = Dict(v => Index(spinone_u1) for v in vertices(g))
    network, env =
        random_state(rng, Float64, g, site_axes; nlayers = 1, trunc = truncrank(4))
    gate = randn_operator(rng, Float64, (site_axes[1], site_axes[2]))
    @test_throws ArgumentError apply_operators(
        [gate], network, env;
        operator_alg = BPApplyGate(; subalgorithm = BufferedBPGateUpdate())
    )
end

@testset "BeliefPropagationEnvironmentPreparation (T=$T)" for T in (Float64, ComplexF64)
    rng = StableRNG(123)
    g = named_path_graph(4)
    site_axes = Dict(v => Index(spinone) for v in vertices(g))
    network, env = random_state(rng, T, g, site_axes; nlayers = 2, trunc = truncrank(4))
    two_site(v1, v2) = randn_operator(rng, T, (site_axes[v1], site_axes[v2]))
    one_site(v) = randn_operator(rng, T, (site_axes[v],))
    gates = [
        two_site(1, 2), two_site(3, 4), two_site(2, 3), one_site(1),
        two_site(1, 2), two_site(3, 4), two_site(2, 3),
    ]
    trunc = truncrank(2)
    maxiter = 3

    count = Ref(0)
    bp_kwargs = (;
        stopping_criterion = (; maxiter),
        message_update_algorithm = CountingMessageUpdate(count),
    )
    beliefpropagation(NormNetwork(network, branamemap(env)), env; bp_kwargs...)
    updates_per_run = count[]
    count[] = 0

    @testset "StopWhenVertexRevisited matches explicit belief propagation" begin
        environment_alg = bp_kwargs
        gated, gated_env = apply_operators(gates, network, env; environment_alg, trunc)
        # Gates 3, 5 and 7 each act on a vertex updated since belief propagation last ran.
        @test count[] == 3 * updates_per_run
        count[] = 0

        reference, reference_env = network, env
        for chunk in (1:2, 3:4, 5:6, 7:7)
            if first(chunk) > 1
                reference_env = beliefpropagation(
                    NormNetwork(reference, branamemap(reference_env)), reference_env;
                    bp_kwargs...
                )
            end
            reference, reference_env =
                apply_operators(gates[chunk], reference, reference_env; trunc)
        end
        count[] = 0
        # The two runs mint different bond names, so compare name-independent quantities.
        @test norm(prod(gated) - prod(reference)) <= 1.0e-12 * norm(prod(reference))
        # Without belief propagation the truncations differ.
        ungated, _ = apply_operators(gates, network, env; trunc)
        @test norm(prod(ungated) - prod(reference)) > 1.0e-6 * norm(prod(reference))
        for edge in edges(gated_env)
            @test norm(gated_env[edge]) ≈ norm(reference_env[edge]) rtol = 1.0e-12
        end
    end

    @testset "StopAfterIteration($k) runs belief propagation every $k gates" for k in 1:3
        when = AI.StopAfterIteration(k)
        environment_alg = (; when, bp_kwargs...)
        apply_operators(gates, network, env; environment_alg, trunc)
        @test count[] == div(length(gates) - 1, k) * updates_per_run
        count[] = 0
    end

    @testset "converged messages name the bond shared with the vertex" begin
        environment_alg =
            BeliefPropagationEnvironmentPreparation(network, env; bp_kwargs...)
        gated, gated_env = apply_operators(gates[1:3], network, env; environment_alg, trunc)
        count[] = 0
        for edge in edges(gated_env)
            message = gated_env[edge]
            @test only(intersect(names(gated[dst(edge)]), names(message))) ==
                only(inputnames(message))
        end
    end
end
