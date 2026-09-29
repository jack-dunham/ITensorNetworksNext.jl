using AlgorithmsInterface: AlgorithmsInterface as AI
using LinearAlgebra: diag, norm
using NamedGraphs: all_edges

default_ctmrg_messages(tn) = Dict(e => ones(Tuple(linkinds(tn, e))) for e in all_edges(tn))
default_ctmrg_messages(nn::NormNetwork) = message_environment(one, nn)

struct CTMRGProblem{Network} <: AI.Problem
    network::Network
end

@kwdef struct CTMRGAlgorithm{
        Faces, Subalgorithm <: AI.Algorithm, StoppingCriterion <: AI.StoppingCriterion,
    } <: AIE.NestedAlgorithm
    faces::Faces
    subalgorithm::Subalgorithm
    stopping_criterion::StoppingCriterion
end

@kwdef mutable struct CTMRGState{
        Substate <: AI.State, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: AIE.NestedState
    substate::Substate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
    delta::Float64 = Inf
end

function AI.initialize_state(
        problem::CTMRGProblem, algorithm::CTMRGAlgorithm; iterate, iteration::Int = 0
    )
    subproblem = CTMRGSweepProblem(problem.network, algorithm.faces)
    substate = AI.initialize_state(subproblem, algorithm.subalgorithm; iterate)
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return CTMRGState(; iteration, stopping_criterion_state, substate)
end

function AI.initialize_state!(
        problem::CTMRGProblem, algorithm::CTMRGAlgorithm, state::CTMRGState;
        iteration::Int = 0
    )
    state.iteration = iteration
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state
    )
    return state
end

function AIE.initialize_subsolve(
        problem::CTMRGProblem, algorithm::CTMRGAlgorithm, state::CTMRGState
    )
    subproblem = CTMRGSweepProblem(problem.network, algorithm.faces)
    return subproblem, algorithm.subalgorithm, state.substate
end

struct CTMRGSweepProblem{Network, Faces} <: AI.Problem
    network::Network
    faces::Faces
end

@kwdef struct CTMRGSweepAlgorithm{Alg, StoppingCriterion <: AI.StoppingCriterion} <:
    AI.Algorithm
    maxdim::Int
    subspace_algorithm::Alg
    stopping_criterion::StoppingCriterion
end

@kwdef mutable struct CTMRGSweepState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: AI.State
    iterate::Iterate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

function AI.initialize_state(
        problem::CTMRGSweepProblem, algorithm::CTMRGSweepAlgorithm;
        iterate, iteration::Int = 0
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return CTMRGSweepState(; iterate, iteration, stopping_criterion_state)
end

function AI.initialize_state!(
        problem::CTMRGSweepProblem, algorithm::CTMRGSweepAlgorithm,
        state::CTMRGSweepState; iteration::Int = 0
    )
    state.iteration = iteration
    AI.initialize_state!(
        problem, algorithm, algorithm.stopping_criterion, state.stopping_criterion_state
    )
    return state
end

function AI.step!(
        problem::CTMRGSweepProblem, algorithm::CTMRGSweepAlgorithm, state::CTMRGSweepState
    )
    f = problem.faces[state.iteration]
    face_update!(
        state.iterate, problem.network, f;
        maxdim = algorithm.maxdim, alg = algorithm.subspace_algorithm
    )
    return state
end

# Records the change over one sweep so `ctmrg` can report whether `tol` was reached.
function AI.step!(problem::CTMRGProblem, algorithm::CTMRGAlgorithm, state::CTMRGState)
    previous = copy(state.substate.iterate)
    invoke(
        AI.step!, Tuple{AI.Problem, AIE.NestedAlgorithm, AI.State}, problem, algorithm,
        state
    )
    state.delta = AIE.iterate_diff(state.substate.iterate, previous)
    return state
end

# Change in the spectrum of each face's eigenvalue corner `c[d_{m-1}]`, divided by its largest
# eigenvalue, whose scale drifts between sweeps without changing Z_B.
function AIE.iterate_diff(env1::CTMEnvironment, env2::CTMEnvironment)
    emb = env1.embedding
    return maximum(eachindex(emb.faces); init = 0.0) do f
        d = emb.faces[f][end - 1]
        function spectrum(env)
            return sort(
                abs.(
                    diag(
                        tomatrix(
                            corner(env, d), (name(bond(env, d)),),
                            (name(bond(env, next_dart(emb, d))),)
                        )
                    )
                )
            )
        end
        s1, s2 = spectrum(env1), spectrum(env2)
        length(s1) == length(s2) || return Inf
        return norm(s1 / last(s1) - s2 / last(s2))
    end
end

"""
    ctmrg(
        tn, emb; maxdim, stopping_criterion::NamedTuple = (; maxiter, tol),
        subspace_algorithm, messages, faces, bp_stopping_criterion
    )

MP-BP environment of `tn` from eig-CTMRG sweeps over the faces of `emb`, started from the
converged BP environment. `stopping_criterion` is required. `faces` is the sweep order over
inner-face indices (default `eachindex(emb.faces)`); `bp_stopping_criterion` (default
`(; maxiter = 100, tol = 1.0e-14)`) controls the initial BP run. Throws if the change over the
last sweep is not below `tol`.
"""
function ctmrg(
        tn, emb::PlanarEmbedding; maxdim::Integer, stopping_criterion::NamedTuple,
        subspace_algorithm = nothing, messages = nothing,
        faces = eachindex(emb.faces),
        bp_stopping_criterion = (; maxiter = 100, tol = 1.0e-14)
    )
    (; maxiter, tol) = stopping_criterion
    messages = isnothing(messages) ? default_ctmrg_messages(tn) : messages
    cache = beliefpropagation(tn, messages; stopping_criterion = bp_stopping_criterion)
    env = ctm_environment(tn, emb, cache)
    alg = select_algorithm(
        invariant_subspace,
        subspace_algorithm,
        Tuple{Matrix{Float64}, Int}
    )
    subalgorithm = CTMRGSweepAlgorithm(;
        maxdim, subspace_algorithm = alg,
        stopping_criterion = AI.StopAfterIteration(length(faces))
    )
    algorithm = CTMRGAlgorithm(;
        faces, subalgorithm,
        stopping_criterion = AI.StopAfterIteration(maxiter) | StopWhenConverged(; tol)
    )
    problem = CTMRGProblem(tn)
    state = AI.initialize_state(problem, algorithm; iterate = env)
    AI.solve!(problem, algorithm, state)
    state.delta < tol || error(
        "`ctmrg` stopped after $(state.iteration) sweeps with `iterate_diff` = $(state.delta), " *
            "not below `tol` = $tol."
    )
    return state.substate.iterate
end
