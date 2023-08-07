module Filtering

export make_observations, filtering_cycles

using Statistics
using LinearAlgebra
using Random

using Distributions
using ProgressMeter

function make_observations(; ensemble, model_true::Function,
                           h, h_hidden=nothing, integrator::Function, Γ,
                           Δt::float_type, window::int_type, n_cycles::int_type,
                           p::int_type, ens_size, p_hidden=0, D=D,
                           t0=0.0) where {float_type<:AbstractFloat,int_type<:Integer}
    trues = Array{float_type}(undef, n_cycles, p)
    ensembles = Array{float_type}(undef, n_cycles, D, ens_size)
    if p_hidden > 0
        trues_hidden = Array{float_type}(undef, n_cycles, p_hidden)
    end
    observations = Array{float_type}(undef, n_cycles, p)
    if Γ !== nothing
        obs_err_dist = MvNormal(Γ)
    end

    t = t0

    for cycle in 1:n_cycles
        E = copy(ensemble)
        for i in 1:ens_size
            E[:, i] = integrator(model_true, E[:, i], t, t + window * Δt, Δt)
        end
        ensembles[cycle, :, :] = E

        hE = [h(E[:, i]) for i in 1:ens_size]
        trues[cycle, :] = mean(hE)
        if p_hidden > 0
            hE_hidden = [h_hidden(E[:, i]) for i in 1:ens_size]
            trues_hidden[cycle, :] = mean(hE_hidden)
        end

        if Γ === nothing
            y = trues[cycle, :]
        else
            y = trues[cycle, :] + rand(obs_err_dist)
        end

        observations[cycle, :] = y
        ensemble = E

        t += window * Δt
    end

    if p_hidden > 0
        return trues, ensembles, trues_hidden, observations
    else
        return trues, ensembles, observations
    end
end

function filtering_cycles(; ensemble::AbstractMatrix{float_type},
                          model::Function, h, observations, integrator::Function,
                          filter_method::Function,
                          ens_size::int_type, Δt::float_type, window::int_type,
                          n_cycles::int_type, model_size::int_type,
                          Γ::AbstractMatrix{float_type},
                          assimilate_obs::Bool=true,
                          t0=0.0, calc_score=false,
                          max_cycle=nothing) where {float_type<:AbstractFloat,
                                                    int_type<:Integer}
    if max_cycle === nothing
        max_cycle = n_cycles
    end

    analyses = Array{float_type}(undef, n_cycles, model_size, ens_size)

    t = t0

    E = ensemble
    @showprogress for cycle in 1:n_cycles
        y = observations[cycle, :]

        if assimilate_obs & (cycle <= max_cycle)
            E_a = filter_method(; E=E, Γ=Γ, h=h, y=y, calc_score=calc_score)
        else
            E_a = E
        end

        analyses[cycle, :, :] = E_a

        for i in 1:ens_size
            integration = integrator(model, E_a[:, i], t, t + window * Δt, Δt; inplace=false)
            E[:, i] = integration[end, :]
        end

        t += window * Δt
    end

    return analyses
end

end