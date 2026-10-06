using .AlgorithmsInterfaceExtensions: AlgorithmsInterfaceExtensions as AIE
using AlgorithmsInterface: AlgorithmsInterface as AI
using Base: @kwdef
using Graphs: dst, src, vertices
using ITensorBase: AbstractITensor, AbstractNamedTensor, ITensor, Index, NamedTensor,
    NamedTensorOperator, apply, inputinds, inputnames, name, names, operator, outputinds,
    rename, sim, state, uniquename, unnamed
using LinearAlgebra: norm, normalize!
using MatrixAlgebraKit: eigh_full, project_hermitian, qr_compact, svd_trunc
using NamedGraphs: boundary_edges, vertextype
using TensorAlgebra.MatrixAlgebra: invsqrth_safe, sqrth_invsqrth_safe, sqrth_safe
using TensorAlgebra: isdual, matricize, twist!, unmatricize

# Asymmetric (Gram) root of a Hermitian positive semidefinite matrix, as the pair
# `(root, inv_root)`: `root' * root == m`, and `inv_root * root` is the identity on `m`'s
# support. `eigh_full` is a square decomposition, so both factors are square.
function message_gauge(m::AbstractMatrix; kwargs...)
    d, u = eigh_full(m)
    return sqrth_safe(d; kwargs...) * u', u * invsqrth_safe(d; kwargs...)
end

# The same root for a named tensor split into `outinds` and `bondinds`, as the pair `(x, y)`:
# `x * ψ` absorbs it into a state tensor `ψ` and `y * ·` un-absorbs it.
#
# `y` inverts `x` as a matrix, and `contract` is that matrix product plus a twist, so the two
# have different identity elements. Twisting `y` on its dual bond axes, the same axes `contract`
# twists, makes it the inverse under `contract`. `twist!` is the identity on storage carrying no
# sector data, so this is a no-op outside the fermionic case.
#
# The message is matricized ket to bra, which keeps the absorbed wavefunction ket-like, and the
# eigendecomposition hands back the rank space already carrying that orientation.
function message_gauge(t::AbstractNamedTensor, outinds, bondinds; kwargs...)
    root, inv_root = message_gauge(matricize(t, outinds, bondinds); kwargs...)
    rankind = Index(axes(root, 1))
    y = unmatricize(inv_root, bondinds, (rankind,))
    twist!(y, filter(isdual, bondinds))
    return unmatricize(root, (rankind,), bondinds), y
end

# A message's output names are the bra side and its input names the ket side, which is the split
# the gauge is taken over. A message is Hermitian only up to numerical noise, and the root needs
# it exactly so.
function message_gauge(m::NamedTensorOperator; kwargs...)
    h = project_hermitian(m)
    return message_gauge(state(h), Tuple(outputinds(h)), Tuple(inputinds(h)); kwargs...)
end

# === Top-level user entry point ===

"""
    apply_operators(operators, state, env; alg=nothing, vertices=nothing, kwargs...)
        -> (state, env)

Apply each operator in `operators` (a sequence of single-tensor or two-tensor
operators) to `state` in turn, updating `env` to reflect each application.
`state` is an `AbstractITensorNetwork`, `env` is a per-edge environment cache
(typically built by `identity_norm_message_env(state)` or one of the related
`*_norm_message_env` constructors), and the returned `(state, env)` pair has
the operators applied. `kwargs` are forwarded to the per-operator algorithm
(`alg`); for the default BP simple-update algorithm these include `trunc`
(forwarded to the SVD that splits a two-site gate back into single-site
tensors) and `normalize`.

`vertices` holds one vertex list per operator, the vertices of `state` that operator acts
on. By default each list is the operator's [`operator_support`](@ref) in `state`, and an
`ArgumentError` is thrown if an input name of an operator is on no tensor of `state`.

See also [`apply_operator`](@ref).
"""
function apply_operators(
        operators, state, env; alg = nothing, vertices = nothing, kwargs...
    )
    algorithm = select_algorithm(
        apply_operators, alg, (operators, state, env); kwargs...
    )
    return apply_operators(algorithm, operators, state, env; vertices)
end

# The `apply_operators` iteration algorithm wraps the per-operator algorithm,
# which is itself resolved via `apply_operator` (overridable with `operator_alg`).
function default_algorithm(
        ::typeof(apply_operators), args::Tuple;
        operator_alg = nothing, environment_alg = nothing, kwargs...
    )
    operators, state, env = args
    # `apply_operator` acts on a single operator, so select on the operator
    # element type, keeping the remaining `(state, env)` argument types.
    # We use types here in case the operator list is empty.
    operator_args = Tuple{eltype(operators), typeof(state), typeof(env)}
    operator_algorithm =
        select_algorithm(apply_operator, operator_alg, operator_args; kwargs...)
    # `apply_operator_environment_preparation` signature (minus the env algorithm):
    # `(operator_algorithm, operators, iteration::Int, iterate, env)`.
    prepare_args = (operator_algorithm, operators, 0, state, env)
    environment_algorithm = select_algorithm(
        apply_operator_environment_preparation, environment_alg, prepare_args
    )
    return ApplyOperatorsAlgorithm(;
        operator_algorithm,
        environment_algorithm,
        stopping_criterion = AI.StopAfterIteration(length(operators))
    )
end

function apply_operators(algorithm, operators, state, env; vertices = nothing)
    isempty(operators) && return copy(state), copy(env)
    if isnothing(vertices)
        vertices = map(operators) do op
            for name in inputnames(op)
                has_dimname(state, name) || throw(
                    ArgumentError(
                        "operator input `$name` is not on any tensor of the network"
                    )
                )
            end
            return operator_support(state, op)
        end
    elseif length(vertices) != length(operators)
        throw(
            ArgumentError(
                "got $(length(vertices)) vertex lists for $(length(operators)) operators"
            )
        )
    end
    problem = ApplyOperatorsProblem(; operators, vertices, init = state)
    return AI.solve(problem, algorithm; iterate = state, env)
end

# === Layer 1: apply_operators iteration ===

@kwdef struct ApplyOperatorsProblem{Ops, Vertices, Init} <: AI.Problem
    operators::Ops
    vertices::Vertices
    init::Init
end

@kwdef struct ApplyOperatorsAlgorithm{
        OperatorAlgorithm,
        EnvironmentAlgorithm,
        StoppingCriterion <: AI.StoppingCriterion,
    } <: AI.Algorithm
    operator_algorithm::OperatorAlgorithm
    environment_algorithm::EnvironmentAlgorithm = NoApplyOperatorEnvironmentPreparation()
    stopping_criterion::StoppingCriterion = AI.StopAfterIteration(0)
end

@kwdef mutable struct ApplyOperatorsState{
        Iterate, Env, StoppingCriterionState <: AI.StoppingCriterionState, EnvironmentState,
    } <: AI.State
    iterate::Iterate
    env::Env
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
    environment_state::EnvironmentState
end

function AI.initialize_state(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperatorsAlgorithm;
        iterate, env, iteration::Int = 0
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    environment_state = initialize_environment_state(
        algorithm.environment_algorithm, problem, algorithm; iterate
    )
    return ApplyOperatorsState(;
        iterate, env, iteration, stopping_criterion_state, environment_state
    )
end

function AI.step!(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperatorsAlgorithm,
        state::ApplyOperatorsState
    )
    # Prepare for the operator application, for example by updating the
    # environments in a path between where the operators are being applied.
    state.iterate, state.env = apply_operator_environment_preparation(
        algorithm.environment_algorithm, problem, algorithm, state
    )
    state.iterate, state.env = apply_operator(
        algorithm.operator_algorithm, problem.operators[state.iteration], state.iterate,
        state.env; vertices = problem.vertices[state.iteration]
    )
    return state
end

function AI.finalize_state!(
        ::ApplyOperatorsProblem, ::ApplyOperatorsAlgorithm, state::ApplyOperatorsState
    )
    return state.iterate, state.env
end

# === Layer 2: environment-preparation strategy ===

# Update the environment (and possibly the factors) before the next operator is
# applied. The full `operators`/`iteration` and `operator_algorithm` are passed so
# a strategy can judge which messages went stale and how much to recompute; it may
# also return regauged/orthogonalized factors. Only the no-op is implemented for
# now (reconvergence policies are follow-up work).
struct NoApplyOperatorEnvironmentPreparation <: AbstractAlgorithm end

function apply_operator_environment_preparation(
        ::NoApplyOperatorEnvironmentPreparation, problem, algorithm, state
    )
    return state.iterate, state.env
end

function initialize_environment_state(
        ::NoApplyOperatorEnvironmentPreparation, problem, algorithm; iterate
    )
    return nothing
end

function default_algorithm(
        ::typeof(apply_operator_environment_preparation), ::Type{<:Tuple}; kwargs...
    )
    return NoApplyOperatorEnvironmentPreparation()
end

"""
    BeliefPropagationEnvironmentPreparation(; when = StopWhenVertexRevisited(), algorithm)
    BeliefPropagationEnvironmentPreparation(network, env; when, kwargs...)

Environment preparation that runs the belief propagation `algorithm` on the environment
before a gate whenever the stopping criterion `when` is met, counting only the gates
applied since belief propagation last ran. The second form builds `algorithm` with
[`beliefpropagation_algorithm`](@ref) from `kwargs`, which must include its
`stopping_criterion`; `apply_operators(...; environment_alg = (; when, kwargs...))` uses it.
"""
struct BeliefPropagationEnvironmentPreparation{
        When <: AI.StoppingCriterion, Algorithm <: BeliefPropagationAlgorithm,
    } <: AbstractAlgorithm
    when::When
    algorithm::Algorithm
end
function BeliefPropagationEnvironmentPreparation(;
        when = StopWhenVertexRevisited(), algorithm
    )
    return BeliefPropagationEnvironmentPreparation(when, algorithm)
end
function BeliefPropagationEnvironmentPreparation(
        network, env; when = StopWhenVertexRevisited(), kwargs...
    )
    factors = message_normnetwork(network, env)
    algorithm = beliefpropagation_algorithm(factors, MessageCache(env); kwargs...)
    return BeliefPropagationEnvironmentPreparation(when, algorithm)
end

# Keyword arguments, from an `environment_alg` `NamedTuple`, select belief propagation.
function default_algorithm(
        ::typeof(apply_operator_environment_preparation), args::Tuple; kwargs...
    )
    isempty(kwargs) && return NoApplyOperatorEnvironmentPreparation()
    _, _, _, network, env = args
    return BeliefPropagationEnvironmentPreparation(network, env; kwargs...)
end

# `iteration` counts the gates applied since belief propagation last ran;
# `operator_index` is the index in `problem.operators` of the next gate.
@kwdef mutable struct BeliefPropagationEnvironmentPreparationState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: AI.State
    iterate::Iterate
    iteration::Int = 0
    operator_index::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

function initialize_environment_state(
        environment_algorithm::BeliefPropagationEnvironmentPreparation, problem, algorithm;
        iterate
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, environment_algorithm.when; iterate
    )
    return BeliefPropagationEnvironmentPreparationState(; iterate, stopping_criterion_state)
end

function apply_operator_environment_preparation(
        environment_algorithm::BeliefPropagationEnvironmentPreparation, problem, algorithm,
        state
    )
    environment_state = state.environment_state
    environment_state.iterate = state.iterate
    environment_state.operator_index = state.iteration
    when = environment_algorithm.when
    if AI.is_finished!(
            problem, algorithm, environment_state, when,
            environment_state.stopping_criterion_state
        )
        state.env = environment_beliefpropagation(
            environment_algorithm, problem, algorithm, state
        )
        environment_state.iteration = 0
        AI.initialize_state!(
            problem, algorithm, when, environment_state.stopping_criterion_state
        )
    end
    environment_state.iteration += 1
    return state.iterate, state.env
end

"""
    environment_beliefpropagation(environment_algorithm, problem, algorithm, state) -> env

The environment after running belief propagation on `state.env` with the norm network of
`state.iterate`, called by [`BeliefPropagationEnvironmentPreparation`](@ref) when its
criterion is met.
"""
function environment_beliefpropagation(
        environment_algorithm::BeliefPropagationEnvironmentPreparation, problem, algorithm,
        state
    )
    bp_problem = BeliefPropagationProblem(message_normnetwork(state.iterate, state.env))
    return AI.solve(
        bp_problem, environment_algorithm.algorithm; iterate = MessageCache(state.env)
    )
end

"""
    StopWhenVertexRevisited()

Stopping criterion for [`BeliefPropagationEnvironmentPreparation`](@ref), met when the next
operator acts on two vertices and one of them was updated since belief propagation last
ran. The vertices of each operator are read from the problem's `vertices`.
"""
struct StopWhenVertexRevisited <: AI.StoppingCriterion end

struct StopWhenVertexRevisitedState{V} <: AI.StoppingCriterionState
    updated::Set{V}
end

function AI.initialize_state(
        ::AI.Problem, ::AI.Algorithm, ::StopWhenVertexRevisited; iterate
    )
    return StopWhenVertexRevisitedState(Set{vertextype(iterate)}())
end

function AI.initialize_state!(
        ::AI.Problem, ::AI.Algorithm, ::StopWhenVertexRevisited,
        st::StopWhenVertexRevisitedState
    )
    empty!(st.updated)
    return st
end

function AI.is_finished(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperatorsAlgorithm,
        state::BeliefPropagationEnvironmentPreparationState, ::StopWhenVertexRevisited,
        st::StopWhenVertexRevisitedState
    )
    vertices = problem.vertices[state.operator_index]
    return length(vertices) == 2 && any(in(st.updated), vertices)
end

# Records the vertices of the previous gate before checking the next one, so the gate
# applied right after belief propagation runs is recorded even though the state was reset.
function AI.is_finished!(
        problem::ApplyOperatorsProblem, algorithm::ApplyOperatorsAlgorithm,
        state::BeliefPropagationEnvironmentPreparationState, c::StopWhenVertexRevisited,
        st::StopWhenVertexRevisitedState
    )
    if state.iteration > 0
        union!(st.updated, problem.vertices[state.operator_index - 1])
    end
    return AI.is_finished(problem, algorithm, state, c, st)
end

AI.indicates_convergence(::StopWhenVertexRevisited) = false
AI.get_reason(::StopWhenVertexRevisited, ::StopWhenVertexRevisitedState) = nothing

# === Layer 3: single-operator strategy ===

abstract type ApplyOperatorAlgorithm <: AbstractAlgorithm end

"""
    apply_operator(operator, state, env; alg=nothing, kwargs...) -> (state, env)

Apply a single `operator` to `state` and return an updated `(state, env)` pair.
Environments are not fully recomputed; only the edges touched by `operator`
are updated (for a two-site gate, the BP simple-update default writes new
messages on the gate edge). For the BP simple-update default algorithm,
`kwargs` accept `trunc` (forwarded to the SVD that splits the gate back into
single-site tensors) and `normalize` (whether to rescale the post-gate state
so the singular-value spectrum stays unit-norm).

See also [`apply_operators`](@ref).
"""
function apply_operator(operator, state, env; alg = nothing, kwargs...)
    algorithm = select_algorithm(apply_operator, alg, (operator, state, env); kwargs...)
    return apply_operator(algorithm, operator, state, env)
end

function apply_operator(algorithm::ApplyOperatorAlgorithm, operator, state, env; kwargs...)
    dest, env_dest = initialize_output(apply_operator!, algorithm, operator, state, env)
    apply_operator!(algorithm, dest, operator, state, env_dest; kwargs...)
    return dest, env_dest
end

# === Default strategy: BPApplyGate ===

@kwdef struct BPApplyGate{Trunc} <: ApplyOperatorAlgorithm
    trunc::Trunc = nothing
    normalize::Bool = false
end

function apply_operator!(
        algorithm::BPApplyGate, dest, operator, state, env;
        vertices = operator_support(state, operator)
    )
    apply_gate_bp!(
        dest, operator, state, env; vertices, algorithm.trunc, algorithm.normalize
    )
    return dest
end

function initialize_output(
        ::typeof(apply_operator!), ::BPApplyGate, operator, state, env
    )
    return copy(state), copy(env)
end

function default_algorithm(::typeof(apply_operator), ::Type{<:Tuple}; kwargs...)
    return BPApplyGate(; kwargs...)
end

# === BP simple-update implementation ===

function apply_gate_bp!(
        dest::AbstractITensorNetwork, op::AbstractITensor,
        state::AbstractITensorNetwork, env;
        vertices = operator_support(state, op), kwargs...
    )
    isempty(vertices) && throw(
        ArgumentError("operator shares no indices with the tensor network")
    )

    N = Val(length(vertices))

    return apply_gate_bp_nsite!(N, dest, op, state, env, vertices; kwargs...)
end

function apply_gate_bp_nsite!(
        ::Val{N}, dest::AbstractITensorNetwork, op::AbstractITensor,
        state::AbstractITensorNetwork, env, vs; kwargs...
    ) where {N}
    return throw(ArgumentError("$N-site gate decomposition not implemented"))
end

function apply_gate_bp_nsite!(
        ::Val{1}, dest::AbstractITensorNetwork, op::AbstractITensor,
        state::AbstractITensorNetwork, env, vertices;
        normalize, kwargs...
    )
    vertex = only(vertices)
    ψv = apply(op, state[vertex])
    if normalize
        sqrt_messages = [
            sqrth_safe(project_hermitian(env[e])) for
                e in boundary_edges(env, vertices; dir = :in)
        ]
        ψv /= norm(foldl((ψ, m) -> apply(m, ψ), sqrt_messages; init = ψv))
    end
    dest[vertex] = ψv
    return dest
end

function apply_gate_bp_nsite!(
        ::Val{2}, dest::AbstractITensorNetwork, op::AbstractITensor,
        state::AbstractITensorNetwork, env, vertices;
        trunc, normalize, bondnames = nothing
    )
    v1, v2 = vertices
    Q_v1, R_v1, invsqrt_messages_v1 = bp_gate_factorize(op, state, env, v1, v2)
    Q_v2, R_v2, invsqrt_messages_v2 = bp_gate_factorize(op, state, env, v2, v1)
    R_v1, R_v2, message_v1v2, message_v2v1 = bp_gate_split(
        op, R_v1, R_v2; trunc, normalize, bondnames
    )
    dest[v1] = bp_gate_restore(Q_v1, R_v1, invsqrt_messages_v1)
    dest[v2] = bp_gate_restore(Q_v2, R_v2, invsqrt_messages_v2)
    env[v1 => v2] = message_v1v2
    env[v2 => v1] = message_v2v1
    return dest
end

"""
    bp_gate_factorize(op, state, env, v, w) -> (Q, R, invsqrt_messages)

Gauge `state[v]` by the square roots of the messages in `env` on every edge into `v`
except `w => v`, and QR-factorize it so that `R` carries the bond to `w` and the names
`state[v]` shares with `op`. `invsqrt_messages` are the inverse square roots that undo
the gauge, for [`bp_gate_restore`](@ref).

`w` need not be a vertex of `state`: the bond is identified as the name `state[v]`
shares with `env[w => v]`.
"""
function bp_gate_factorize(op::AbstractITensor, state, env, v, w)
    edges_in = [e for e in boundary_edges(env, [v]; dir = :in) if src(e) != w]
    roots = [sqrth_invsqrth_safe(project_hermitian(env[e])) for e in edges_in]
    sqrt_messages, invsqrt_messages = first.(roots), last.(roots)
    ψ = foldl((ψ, m) -> apply(m, ψ), sqrt_messages; init = state[v])
    bondname = only(intersect(names(state[v]), names(env[w => v])))
    Q, R = qr_compact(ψ, setdiff(names(ψ), [bondname], names(op)))
    return Q, R, invsqrt_messages
end

"""
    bp_gate_split(op, R_v1, R_v2; trunc, normalize, bondnames = nothing)
        -> (R_v1, R_v2, message_v1v2, message_v2v1)

Apply the two-site `op` to `R_v1 * R_v2`, truncate with `svd_trunc(...; trunc)`
(normalizing the singular values if `normalize`), and split the result back into two
factors by the square root of the singular values. Returns the new factors and the
`R†R` messages `v1 => v2` and `v2 => v1` on the new bond.

The new factors share one bond name, which is the input name of both messages; the
messages' output name appears in neither factor. `bondnames = (input, output)` sets
these two names; by default they are the names `svd_trunc` mints.
"""
function bp_gate_split(
        op::AbstractITensor, R_v1::AbstractITensor, R_v2::AbstractITensor;
        trunc, normalize, bondnames = nothing
    )
    op_R_v1v2 = apply(op, R_v1 * R_v2)
    U_v1, S, U_v2 = svd_trunc(op_R_v1v2, setdiff(names(R_v1), names(R_v2)); trunc)
    if !isnothing(bondnames)
        name_u, name_v = names(S)
        name_v1, name_v2 = bondnames
        U_v1 = rename(U_v1, name_u => name_v1)
        S = rename(S, name_u => name_v1, name_v => name_v2)
        U_v2 = rename(U_v2, name_v => name_v2)
    end

    normalize && normalize!(S)

    name_v1, name_v2 = names(S)
    sqrt_S = sqrth_safe(S, (name_v1,), (name_v2,); atol = 0, rtol = 0)
    R_v1 = rename(U_v1 * sqrt_S, name_v2 => name_v1)
    R_v2 = sqrt_S * U_v2

    message_v1v2 = operator(
        rename(conj(R_v1), name_v1 => name_v2) * R_v1, (name_v2,), (name_v1,)
    )
    message_v2v1 = operator(
        rename(conj(R_v2), name_v1 => name_v2) * R_v2, (name_v2,), (name_v1,)
    )
    return R_v1, R_v2, message_v1v2, message_v2v1
end

"""
    bp_gate_restore(Q, R, invsqrt_messages)

The vertex tensor `Q * R` with the gauge of [`bp_gate_factorize`](@ref) undone by
applying `invsqrt_messages`.
"""
function bp_gate_restore(Q::AbstractITensor, R::AbstractITensor, invsqrt_messages)
    return foldl((ψ, m) -> apply(m, ψ), invsqrt_messages; init = Q * R)
end
