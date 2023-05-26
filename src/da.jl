module DA

export make_observations, da_cycles, compute_stats, gaspari_cohn

using Statistics
using LinearAlgebra
using Random

using Distributions
using ProgressMeter
using PyCall

mutable struct DA_Info
    forecasts
    analyses
    P_hist
end

mutable struct Metrics
    errs
    errs_fcst
    crps
    crps_fcst
    spread
    spread_fcst
end

function gaspari_cohn(r)
    if 0 <= r < 1
        G = 1 - 5/3*r^2 + 5/8*r^3 + 1/2*r^4 - 1/4*r^5
    elseif 1 <= r < 2
        G = 4 - 5*r + 5/3*r^2 + 5/8*r^3 - 1/2*r^4 + 1/12*r^5 - 2/(3*r)
    elseif r >= 2
        G = 0
    end
    return G
end

function gaspari_cohn_localization(c, D; cyclic=false)
    localization = zeros(D, D)
    for i=1:D
        for j=1:i
            if cyclic
                r = min(mod(i - j, 0:D), mod(j - i, 0:D))/c
            else
                r = abs(i - j)
            end
            localization[i, j] = DA.gaspari_cohn(r)
        end
    end
    return Symmetric(localization, :L)
end

xskillscore = pyimport("xskillscore")
xarray = pyimport("xarray")

function make_observations(; ensemble, model_true::Function,
                             H_true, H_hidden=nothing, integrator::Function, R::AbstractMatrix{float_type},
                             Δt::float_type, window::int_type, n_cycles::int_type, outfreq::int_type,
                             p::int_type, ens_size, p_hidden=0, D=D, t0=0.0) where {float_type<:AbstractFloat, int_type<:Integer}
    trues = Array{float_type}(undef, n_cycles, p)
    ensembles = Array{float_type}(undef, n_cycles, D, ens_size)
    if p_hidden > 0
        trues_hidden = Array{float_type}(undef, n_cycles, p_hidden)
    end
    covariances = Array{Array{float_type}}(undef, n_cycles)
    observations = Array{float_type}(undef, n_cycles, size(R, 1))
    obs_err_dist = MvNormal(R)

    t = t0

    for cycle=1:n_cycles
        E = copy(ensemble)
        Threads.@threads for i=1:ens_size
            E[:, i] = integrator(model_true, E[:, i], t, t + window*outfreq*Δt, Δt)
        end
        ensembles[cycle, :, :] = E

        HE = [H_true(E[:, i]) for i=1:ens_size]
        trues[cycle, :] = mean(HE)
        if p_hidden > 0
            HE_hidden = [H_hidden(E[:, i]) for i=1:ens_size]
            trues_hidden[cycle, :] = mean(HE_hidden)
        end
        covariances[cycle] = cov(HE)

        y = trues[cycle, :] + rand(obs_err_dist)

        observations[cycle, :] = y
        ensemble = E

        t += window*outfreq*Δt
        #x_true = integrator(model_true, x_true, t, t + window*outfreq*Δt, Δt)
    end

    if p_hidden > 0
        return trues, ensembles, trues_hidden, observations, mean(covariances)
    else
        return trues, ensembles, observations, mean(covariances)
    end
end

function da_cycles(; ensemble::AbstractMatrix{float_type},
                     model::Function, H, H_linear=nothing, observations, integrator::Function,
                     da_method::Function, localization,
                     ens_size::int_type, Δt::float_type, window::int_type,
                     n_cycles::int_type, outfreq::int_type, model_size::int_type,
                     R::AbstractMatrix{float_type},
                     assimilate_obs::Bool=true,
                     save_P_hist::Bool=false,
                     prev_analyses::Union{AbstractArray{float_type}, Nothing}=nothing,
                     leads::int_type=1, t0=0.0, calc_score=true, inflation=1.0, max_cycle=nothing) where {float_type<:AbstractFloat, int_type<:Integer}
    if max_cycle === nothing
        max_cycle = n_cycles
    end
    save_fcsts = false
    R_inv = inv(R)

    if save_P_hist
        P_hist = Array{Matrix{float_type}}(undef, n_cycles)
    else
        P_hist = nothing
    end

    if save_fcsts
        forecasts = Array{float_type}(undef, window*n_cycles, model_size, ens_size)
    else
        forecasts = nothing
    end
    analyses = Array{float_type}(undef, n_cycles, model_size, ens_size)

    t = t0

    @showprogress for cycle=1:n_cycles
        y = observations[cycle, :]

        lead = mod(cycle, leads)

        E = ensemble

        P_p = Symmetric(cov(E'))

        if save_P_hist
            P_hist[cycle] = P_p
        end

        if assimilate_obs & (mod(cycle, leads) == 0) & (cycle <= max_cycle)
            E_a = E
            for i=1:1
                E_a = da_method(E=E_a, R=R, R_inv=R_inv, H=H, H_linear=H_linear, y=y, localization=localization, calc_score=calc_score, Δt=Δt, inflation=inflation,
                                model=model)
            end
        else
            E_a = E
        end

        analyses[cycle, :, :] = E_a

        if (prev_analyses !== nothing) & (mod(cycle, leads) == 0)
            E = prev_analyses[cycle, :, :]
        else
            E = E_a
        end

        #if (!assimilate_obs) | (assimilate_obs & mod(cycle, leads) != 0)
            Threads.@threads for i=1:ens_size
                integration = integrator(model, E[:, i], t, t + window*outfreq*Δt, Δt, inplace=false)
                if save_fcsts
                    forecasts[(cycle-1)*window+1:cycle*window, :, i] = integration
                end
                E[:, i] = integration[end, :]
            end
        #end

        ensemble = E

        t += window*outfreq*Δt
    end

    return DA_Info(forecasts, analyses, P_hist)
end

function compute_stats(; da_info::DA_Info, trues::AbstractMatrix{float_type}) where {float_type<:AbstractFloat}
    forecasts = da_info.forecasts
    analyses = da_info.analyses
    n_cycles = size(forecasts, 1)
    model_size = size(forecasts, 2)

    errs = Array{float_type}(undef, n_cycles, model_size)
    errs_fcst = Array{float_type}(undef, n_cycles, model_size)
    crps = Array{float_type}(undef, n_cycles)
    crps_fcst = Array{float_type}(undef, n_cycles)
    spread = Array{float_type}(undef, n_cycles)
    spread_fcst = Array{float_type}(undef, n_cycles)

    for cycle=1:n_cycles
        true_array = xarray.DataArray(data=trues[cycle, :], dims=["dim"])

        E_corr_array = xarray.DataArray(data=analyses[cycle, :, :], dims=["dim", "member"])
        crps[cycle] = xskillscore.crps_ensemble(true_array, E_corr_array).values[1]

        spread[cycle] = mean(std(analyses[cycle, :, :], dims=2))

        errs[cycle, :] = mean(analyses[cycle, :, :], dims=2) - trues[cycle, :]

        errs_fcst[cycle, :] = mean(forecasts[cycle, :, :], dims=2) - trues[cycle, :]

        E_corr_fcst_array = xarray.DataArray(data=forecasts[cycle, :, :], dims=["dim", "member"])
        crps_fcst[cycle] = xskillscore.crps_ensemble(true_array, E_corr_fcst_array).values[1]
        spread_fcst[cycle] = mean(std(forecasts[cycle, :, :], dims=2))
    end

    return Metrics(errs, errs_fcst, crps, crps_fcst, spread, spread_fcst)
end

end