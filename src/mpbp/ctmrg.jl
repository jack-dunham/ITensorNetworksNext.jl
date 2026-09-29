using AlgorithmsInterface: AlgorithmsInterface as AI
using ITensorBase: unnamed
using LinearAlgebra: diag, norm
using NamedGraphs: all_edges

default_ctmrg_messages(tn) = Dict(e => ones(Tuple(linkinds(tn, e))) for e in all_edges(tn))
default_ctmrg_messages(nn::NormNetwork) = message_environment(one, nn)

struct CTMRGProblem{Network} <: AI.Problem
    network::Network
end

@kwdef struct CTMRGAlgorithm{Faces, Alg, StoppingCriterion <: AI.StoppingCriterion} <:
    AI.Algorithm
    faces::Faces
    maxdim::Int
    subspace_algorithm::Alg
    stopping_criterion::StoppingCriterion
end

@kwdef mutable struct CTMRGState{
        Iterate, StoppingCriterionState <: AI.StoppingCriterionState,
    } <: AI.State
    iterate::Iterate
    iteration::Int = 0
    stopping_criterion_state::StoppingCriterionState
end

function AI.initialize_state(
        problem::CTMRGProblem, algorithm::CTMRGAlgorithm; iterate, iteration::Int = 0
    )
    stopping_criterion_state = AI.initialize_state(
        problem, algorithm, algorithm.stopping_criterion; iterate
    )
    return CTMRGState(; iterate, iteration, stopping_criterion_state)
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

function AI.step!(problem::CTMRGProblem, algorithm::CTMRGAlgorithm, state::CTMRGState)
    for f in algorithm.faces
        face_update!(
            state.iterate, problem.network, f;
            maxdim = algorithm.maxdim, alg = algorithm.subspace_algorithm
        )
    end
    return state
end

# Change in the spectrum of each face's eigenvalue corner `c[d_{m-1}]`, divided by its largest
# eigenvalue, whose scale drifts between sweeps without changing Z_B.
function AIE.iterate_diff(env1::CTMEnvironment, env2::CTMEnvironment)
    return maximum(env1.embedding.faces; init = 0.0) do ds
        s1, s2 = corner_spectrum(env1, ds[end - 1]), corner_spectrum(env2, ds[end - 1])
        length(s1) == length(s2) || return Inf
        return norm(s1 / last(s1) - s2 / last(s2))
    end
end
corner_spectrum(env, d) = sort(abs.(diag(unnamed(corner(env, d)))))

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
    algorithm = CTMRGAlgorithm(;
        faces, maxdim, subspace_algorithm = alg,
        stopping_criterion = AI.StopAfterIteration(maxiter) | StopWhenConverged(; tol)
    )
    problem = CTMRGProblem(tn)
    state = AI.initialize_state(problem, algorithm; iterate = env)
    AI.solve!(problem, algorithm, state)
    convergence = last(state.stopping_criterion_state.criteria_states)
    convergence.at_iteration ≥ 0 || error(
        "`ctmrg` stopped after $(state.iteration) sweeps with `iterate_diff` = " *
            "$(convergence.delta), not below `tol` = $tol."
    )
    return state.iterate
end
