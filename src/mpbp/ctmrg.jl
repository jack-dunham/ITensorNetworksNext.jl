using AlgorithmsInterface: AlgorithmsInterface as AI
using ITensorBase: unnamed
using LinearAlgebra: eigvals, norm
using NamedGraphs: all_edges

default_ctmrg_messages(tn) = Dict(e => ones(Tuple(linkinds(tn, e))) for e in all_edges(tn))
default_ctmrg_messages(nn::NormNetwork) = message_environment(one, nn)

struct CTMRGProblem{Network} <: AI.Problem
    network::Network
end

@kwdef struct FaceUpdate{Alg} <: AbstractAlgorithm
    maxdim::Int
    subspace_algorithm::Alg
    frozen::Bool = false
    align::Bool = false
end

function AIE.update!(update::FaceUpdate, env, problem::CTMRGProblem, face)
    return face_update!(
        env,
        problem.network,
        face;
        maxdim = update.maxdim,
        alg = update.subspace_algorithm,
        frozen = update.frozen,
        align = update.align
    )
end

# Change in the eigenvalues of each face's eigenvalue corner `c[d_{m-1}]`, divided by the
# largest, whose scale drifts between sweeps without changing Z_B.
function AIE.iterate_diff(env1::CTMEnvironment, env2::CTMEnvironment)
    return maximum(env1.embedding.faces; init = 0.0) do face
        spectrum1 = corner_spectrum(env1, face[end - 1])
        spectrum2 = corner_spectrum(env2, face[end - 1])
        length(spectrum1) == length(spectrum2) || return Inf
        return norm(spectrum1 / last(spectrum1) - spectrum2 / last(spectrum2))
    end
end
corner_spectrum(env, edge) = sort(abs.(eigvals(unnamed(corner(env, edge)))))

"""
    ctmrg(
        tn, embedding; maxdim, stopping_criterion::NamedTuple = (; maxiter, tol),
        subspace_algorithm, messages, faces, bp_stopping_criterion
    )

MP-BP environment of `tn` from eig-CTMRG sweeps over the faces of `embedding`, started from the
converged BP environment. `stopping_criterion` is required. `faces` is the sweep order over
inner-face indices (default `eachindex(embedding.faces)`); `bp_stopping_criterion` (default
`(; maxiter = 100, tol = 1.0e-14)`) controls the initial BP run. Throws if the change over the
last sweep is not below `tol`.
"""
function ctmrg(
        tn, embedding::PlanarEmbedding; maxdim::Integer, stopping_criterion::NamedTuple,
        subspace_algorithm = nothing, messages = nothing,
        faces = eachindex(embedding.faces),
        bp_stopping_criterion = (; maxiter = 100, tol = 1.0e-14)
    )
    (; maxiter, tol) = stopping_criterion
    messages = isnothing(messages) ? default_ctmrg_messages(tn) : messages
    cache = beliefpropagation(tn, messages; stopping_criterion = bp_stopping_criterion)
    env = ctm_environment(tn, embedding, cache)

    alg = select_algorithm(
        invariant_subspace,
        subspace_algorithm,
        Tuple{Matrix{Float64}, Int}
    )
    sweep = SweepAlgorithm(;
        schedule = faces,
        update = FaceUpdate(; maxdim, subspace_algorithm = alg)
    )
    algorithm = IterateUntilConverged(;
        subalgorithm = sweep,
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
