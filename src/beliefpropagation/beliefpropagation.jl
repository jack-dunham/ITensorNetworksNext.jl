using .AlgorithmsInterfaceExtensions:
    AlgorithmsInterfaceExtensions as AIE, StopWhenConverged, iterate_diff
using AlgorithmsInterface: AlgorithmsInterface as AI
using DataGraphs: edge_data
using Graphs: AbstractEdge, edges, edgetype, has_edge, vertices
using ITensorBase: AbstractITensor, operator, state
using LinearAlgebra: norm, normalize, tr
using NamedGraphs: forest_cover_edge_sequence, subgraph
using TensorAlgebra: TensorOperationsContract

# === Top-level user entry point ===

default_beliefpropagation_edges(graph) = forest_cover_edge_sequence(graph)

select_beliefpropagation_stopping_criterion(c::AI.StoppingCriterion) = c
function select_beliefpropagation_stopping_criterion(::Nothing)
    return throw(
        ArgumentError(
            "`stopping_criterion` must be specified, e.g.\n" *
                "  `stopping_criterion = (; maxiter = 10)`,\n" *
                "  `stopping_criterion = (; maxiter = 10, tol = 1.0e-10)`, or\n" *
                "  `stopping_criterion = AI.StopAfterIteration(10) | StopWhenConverged(1.0e-10)`."
        )
    )
end
function select_beliefpropagation_stopping_criterion(kwargs::NamedTuple)
    return select_beliefpropagation_stopping_criterion(; kwargs...)
end
function select_beliefpropagation_stopping_criterion(;
        maxiter = nothing, tol = nothing, kwargs...
    )
    if !isempty(kwargs)
        throw(
            ArgumentError(
                "Unrecognized `stopping_criterion` kwargs: $(keys(kwargs)). " *
                    "Supported: `maxiter`, `tol`."
            )
        )
    end
    if isnothing(maxiter) && isnothing(tol)
        throw(
            ArgumentError("At least one of `maxiter` or `tol` must be specified.")
        )
    end
    criterion = nothing
    if !isnothing(maxiter)
        criterion = AI.StopAfterIteration(maxiter)
    end
    if !isnothing(tol)
        converged = StopWhenConverged(; tol)
        criterion = isnothing(criterion) ? converged : criterion | converged
    end
    return criterion
end

"""
    beliefpropagation(factors, messages; edges, stopping_criterion, message_update_algorithm) -> MessageCache

Run belief propagation on the factor graph `factors`, starting from
`messages` (a dictionary keyed by directed edges). Returns the converged
`MessageCache`. `edges` is the sweep schedule (defaults to a forest-cover
edge sequence). `stopping_criterion` is required and accepts a
`NamedTuple` shorthand (`(; maxiter)`, `(; tol)`, `(; maxiter, tol)`) or
an explicit `AlgorithmsInterface.StoppingCriterion`.
`message_update_algorithm` controls how a single message is recomputed
from its incoming neighbours.
"""
function beliefpropagation(
        factors, messages;
        edges = default_beliefpropagation_edges(factors),
        stopping_criterion = nothing,
        message_update_algorithm = nothing
    )
    problem = BeliefPropagationProblem(factors)
    cache = MessageCache(messages)

    # No concrete `edge` value here, so the args tuple uses `edgetype(factors)`.
    message_update_algorithm = select_algorithm(
        message_update!,
        message_update_algorithm,
        Tuple{typeof(cache), typeof(factors), edgetype(factors)}
    )
    subalgorithm = BeliefPropagationSweepAlgorithm(;
        message_update_algorithm,
        stopping_criterion = AI.StopAfterIteration(length(edges))
    )
    stopping_criterion = select_beliefpropagation_stopping_criterion(stopping_criterion)
    algorithm = BeliefPropagationAlgorithm(; edges, subalgorithm, stopping_criterion)

    return AI.solve(problem, algorithm; iterate = cache) # -> typeof(cache)
end

# === Layer 1: BP outer loop (iterative) ===

struct BeliefPropagationProblem{Factors} <: AI.Problem
    factors::Factors
end

@kwdef struct BeliefPropagationAlgorithm{
        Edges,
        Subalgorithm <: AI.Algorithm,
        StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.NestedAlgorithm
    edges::Edges
    subalgorithm::Subalgorithm
    stopping_criterion::StoppingCriterion
end

@kwdef mutable struct BeliefPropagationState{
        Substate <: AI.State, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: AIE.NestedState
    substate::Substate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

function AI.initialize_state(
        problem::BeliefPropagationProblem,
        algorithm::BeliefPropagationAlgorithm;
        iterate, iteration::Int = 0
    )
    subproblem = BeliefPropagationSweepProblem(problem.factors, algorithm.edges)
    substate = AI.initialize_state(subproblem, algorithm.subalgorithm; iterate)
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return BeliefPropagationState(; iteration, stopping_criterion_state, substate)
end

function AI.initialize_state!(
        problem::BeliefPropagationProblem,
        algorithm::BeliefPropagationAlgorithm,
        state::BeliefPropagationState;
        iteration::Int = 0
    )
    state.iteration = iteration
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state
    )
    return state
end

function AIE.initialize_subsolve(
        problem::BeliefPropagationProblem,
        algorithm::BeliefPropagationAlgorithm,
        state::BeliefPropagationState
    )
    subproblem = BeliefPropagationSweepProblem(problem.factors, algorithm.edges)
    return subproblem, algorithm.subalgorithm, state.substate
end

# === Layer 2: one sweep over edges (iterative) ===

struct BeliefPropagationSweepProblem{Factors, Edges} <: AI.Problem
    factors::Factors
    edges::Edges
end

@kwdef struct BeliefPropagationSweepAlgorithm{
        MessageUpdateAlgorithm,
        StoppingCriterion <: AI.StoppingCriterion,
    } <: AI.Algorithm
    message_update_algorithm::MessageUpdateAlgorithm = SimpleMessageUpdate()
    stopping_criterion::StoppingCriterion
end

@kwdef mutable struct BeliefPropagationSweepState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: AI.State
    iterate::Iterate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

function AI.initialize_state(
        problem::BeliefPropagationSweepProblem,
        algorithm::BeliefPropagationSweepAlgorithm;
        iterate, iteration::Int = 0
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return BeliefPropagationSweepState(; iterate, iteration, stopping_criterion_state)
end

function AI.initialize_state!(
        problem::BeliefPropagationSweepProblem,
        algorithm::BeliefPropagationSweepAlgorithm,
        state::BeliefPropagationSweepState;
        iteration::Int = 0
    )
    state.iteration = iteration
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state
    )
    return state
end

function AI.step!(
        problem::BeliefPropagationSweepProblem,
        algorithm::BeliefPropagationSweepAlgorithm,
        state::BeliefPropagationSweepState
    )
    edge = problem.edges[state.iteration]
    message_update!(
        algorithm.message_update_algorithm, state.iterate, problem.factors, edge
    )
    return state
end

# === Layer 3: single-edge message update strategy ===

# Strategy interface: a `MessageUpdateAlgorithm` defines how a single
# message is computed and written back into the message store. Plug in a
# new strategy by subtyping `MessageUpdateAlgorithm` and overloading
# `message_update!(strategy, cache, factors, edge)`.
abstract type MessageUpdateAlgorithm <: AbstractAlgorithm end

function message_update! end

# `args` tuple mirrors the `message_update!(cache, factors, edge)` call shape.
function default_algorithm(::typeof(message_update!), ::Type{<:Tuple}; kwargs...)
    return SimpleMessageUpdate(; kwargs...)
end

# Convenience entry: pick the strategy via `select_algorithm`
# (accepts either `alg = ::MessageUpdateAlgorithm` / `::NamedTuple`, or flat
# kwargs forwarded to the default algorithm), then dispatch.
function message_update!(cache, factors, edge; alg = nothing, kwargs...)
    return message_update!(
        select_algorithm(message_update!, alg, (cache, factors, edge); kwargs...),
        cache, factors, edge
    )
end

@kwdef struct SimpleMessageUpdate{ContractionAlg} <: MessageUpdateAlgorithm
    normalize::Bool = true
    contraction_alg::ContractionAlg = Exact()
end

# Contract the incoming messages into the source factor to form the (unnormalized) new message on
# `edge`.
function updated_message(algorithm::SimpleMessageUpdate, cache, factors, edge)
    messages = collect(incoming_messages(cache, edge))
    return contract_network(
        [messages; [factors[src(edge)]]];
        alg = algorithm.contraction_alg
    )
end

# Single-layer network: the message is a plain bond vector, normalized by its entrywise sum.
function message_update!(algorithm::SimpleMessageUpdate, cache, factors, edge)
    new_message = updated_message(algorithm, cache, factors, edge)
    if algorithm.normalize
        message_norm = sum(new_message)
        iszero(message_norm) || (new_message /= message_norm)
    end
    cache[edge] = new_message
    return cache
end

# Bilinear-form network: the message is a doubled (ket/bra) bond operator. Contracting a plain
# vertex factor with the incoming messages leaves the surviving bond legs dangling, so assign the
# bra/ket pairing the network gives this edge (the same convention as `similar_message_environment`).
# On a `NormNetwork` the message is positive semidefinite and its trace is a positive normalization.
function bilinearform_message_update!(
        algorithm, cache, factors::AbstractBilinearFormNetwork, edge
    )
    new_tensor = updated_message(algorithm, cache, factors, edge)
    branames = linknames(branetwork(factors), edge)
    ketnames = linknames(ketnetwork(factors), edge)
    new_message = operator(new_tensor, branames, ketnames)
    if algorithm.normalize
        message_norm = tr(new_message)
        iszero(message_norm) || (new_message /= message_norm)
    end
    cache[edge] = new_message
    return cache
end

function message_update!(
        algorithm::SimpleMessageUpdate, cache, factors::AbstractBilinearFormNetwork, edge
    )
    return bilinearform_message_update!(algorithm, cache, factors, edge)
end

"""
    BlockedMessageUpdate(; normalize = true, nblocks = nothing, workspace_limit = nothing,
                           contract_alg = TensorOperationsContract())

Message update for a `NormNetwork` or a `QuadraticFormNetwork` that splits the outgoing ket leg of each message into `nblocks`
column blocks of near-equal length, so each intermediate is about `1 / nblocks` of the ket. An
`nblocks` larger than the leg's length gives one column per block, and `nblocks = nothing` chooses
it per message from the contraction backend and the ket's size. The default `contract_alg` requires
TensorOperations to be loaded. Every contraction runs with `contract_alg`, and with a
`TensorOperationsContract` intermediates are allocated and freed through its allocator.
`workspace_limit` is reserved and must be `nothing`.
"""
@kwdef struct BlockedMessageUpdate{ContractAlg} <: MessageUpdateAlgorithm
    normalize::Bool = true
    nblocks::Union{Nothing, Int} = nothing
    workspace_limit::Nothing = nothing
    contract_alg::ContractAlg = TensorOperationsContract()
    function BlockedMessageUpdate(normalize, nblocks, workspace_limit, contract_alg)
        isnothing(nblocks) || nblocks isa Integer && nblocks > 0 ||
            throw(
            ArgumentError(
                "`nblocks` must be `nothing` or a positive integer, got $nblocks."
            )
        )
        isnothing(workspace_limit) || throw(
            ArgumentError("`workspace_limit` is not supported yet and must be `nothing`.")
        )
        return new{typeof(contract_alg)}(normalize, nblocks, workspace_limit, contract_alg)
    end
end

"""
    default_nblocks(algorithm::BlockedMessageUpdate, ket::AbstractArray, χ::Integer) -> Int
    default_nblocks(backend, ketbytes::Integer, χ::Integer) -> Int

The number of column blocks `algorithm` splits a leg of length `χ` of the ket array `ket` into;
the kernel caps it at `χ`. The first form returns `algorithm.nblocks` when it is set; for `nothing`
and a `TensorOperationsContract` it finds the TensorOperations backend `algorithm.contract_alg`
contracts with and calls the second, which a backend overloads, and otherwise returns 1. The
second form is defined by the TensorOperations extension: 1 (the whole leg) by default, and on
cuTENSOR about 16 blocks, each at least 4 MiB and at most 64 columns.
"""
function default_nblocks end
function default_nblocks(algorithm::BlockedMessageUpdate, ket::AbstractArray, χ::Integer)
    return @something algorithm.nblocks 1
end

# Rejects operands whose element types the backend `alg` contracts with cannot mix, before any
# contraction runs. The TensorOperations extension overloads it for cuTENSOR.
check_element_types(alg, tensors) = nothing

function message_update!(
        algorithm::BlockedMessageUpdate, cache, factors::AbstractBilinearFormNetwork, edge
    )
    return bilinearform_message_update!(algorithm, cache, factors, edge)
end

# === `iterate_diff` for `MessageCache` (used by `AIE.StopWhenConverged`) ===

function AIE.iterate_diff(cache1::MessageCache, cache2::MessageCache)
    return maximum(edges(cache1)) do edge
        m1 = cache1[edge]
        m2 = cache2[edge]
        return 1 - abs2(LinearAlgebra.dot(normalize(m1), normalize(m2)))
    end
end
