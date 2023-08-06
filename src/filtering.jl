module Filtering

export make_observations, filtering_cycles

using Statistics
using LinearAlgebra
using Random

using Distributions
using ProgressMeter

function make_observations(; ensemble, model_true::Function,
                             H_true, H_hidden=nothing, integrator::Function, Γ::AbstractMatrix{float_type},
                             Δt::float_type, window::int_type, n_cycles::int_type, outfreq::int_type,
                             p::int_type, ens_size, p_hidden=0, D=D, t0=0.0) where {float_type<:AbstractFloat, int_type<:Integer}
    trues = Array{float_type}(undef, n_cycles, p)
    ensembles = Array{float_type}(undef, n_cycles, D, ens_size)
    if p_hidden > 0
        trues_hidden = Array{float_type}(undef, n_cycles, p_hidden)
    end
    observations = Array{float_type}(undef, n_cycles, size(Γ, 1))
    obs_err_dist = MvNormal(Γ)

    t = t0

    for cycle=1:n_cycles
        E = copy(ensemble)
        for i=1:ens_size
            E[:, i] = integrator(model_true, E[:, i], t, t + window*outfreq*Δt, Δt)
        end
        ensembles[cycle, :, :] = E

        HE = [H_true(E[:, i]) for i=1:ens_size]
        trues[cycle, :] = mean(HE)
        if p_hidden > 0
            HE_hidden = [H_hidden(E[:, i]) for i=1:ens_size]
            trues_hidden[cycle, :] = mean(HE_hidden)
        end

        y = trues[cycle, :] + rand(obs_err_dist)

        observations[cycle, :] = y
        ensemble = E

        t += window*outfreq*Δt
    end

    if p_hidden > 0
        return trues, ensembles, trues_hidden, observations
    else
        return trues, ensembles, observations
    end
end

function filtering_cycles(; ensemble::AbstractMatrix{float_type},
                     model::Function, H, observations, integrator::Function,
                     filter::Function,
                     ens_size::int_type, Δt::float_type, window::int_type,
                     n_cycles::int_type, outfreq::int_type, model_size::int_type,
                     Γ::AbstractMatrix{float_type},
                     assimilate_obs::Bool=true,
                     t0=0.0, calc_score=false, max_cycle=nothing) where {float_type<:AbstractFloat, int_type<:Integer}
    if max_cycle === nothing
        max_cycle = n_cycles
    end
    Γ_inv = inv(Γ)

    analyses = Array{float_type}(undef, n_cycles, model_size, ens_size)

    t = t0

    @showprogress for cycle=1:n_cycles
        y = observations[cycle, :]

        E = ensemble

        if assimilate_obs & (cycle <= max_cycle)
            E_a = filter(E=E_a, Γ=Γ, Γ_inv=Γ_inv, H=H, y=y, calc_score=calc_score, Δt=Δt,
                            model=model)
        else
            E_a = E
        end

        analyses[cycle, :, :] = E_a

        for i=1:ens_size
            integration = integrator(model, E[:, i], t, t + window*outfreq*Δt, Δt, inplace=false)
            E[:, i] = integration[end, :]
        end

        ensemble = E

        t += window*outfreq*Δt
    end

    return analyses
end

end