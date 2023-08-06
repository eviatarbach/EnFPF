using Statistics
using LinearAlgebra
using Random

using Distributions

import EnFPF.Filtering
import EnFPF.Filters
import EnFPF.Models
import EnFPF.Integrators
import EnFPF.Metrics

Random.seed!(10)

D = 3
model = Models.lorenz63_na

p = 9
ens_size = 100
ens_obs_size = 100
model_size = D
integrator = Integrators.rk4
filter_method = Filters.enfpf

x0 = randn(D)
t0 = 0.0
Δt = 0.05
transient = 2000
x = integrator(model, x0, t0, transient * Δt, Δt; inplace=false)

n_cycles = 400

ens_err = Symmetric(diagm(0.25 * ones(D)))

x0 = x[end, :]

window = 1
max_cycle = nothing

ensemble = x0 .+ 0.25 * randn(D, ens_size)
ensemble_obs = x[1000, :] .+ 0.25 * randn(D, ens_obs_size)

h(v) = vcat([v .^ i for i in 1:3]...)

long_n_cycles = 10000

true_states, ensembles, observations = Filtering.make_observations(; ensemble=ensemble_obs,
                                                                   model_true=model, h=h,
                                                                   integrator=integrator,
                                                                   Γ=nothing, Δt=Δt,
                                                                   window=window,
                                                                   n_cycles=long_n_cycles,
                                                                   p=p,
                                                                   ens_size=ens_obs_size,
                                                                   D=D, t0=0.0)

t0 = (long_n_cycles - n_cycles) * window * Δt

Γ = cov(observations; dims=1) / 5
obs_err_dist = MvNormal(Γ)

true_states, ensembles, observations = Filtering.make_observations(; ensemble=ensemble_obs,
                                                                   model_true=model, h=h,
                                                                   integrator=integrator,
                                                                   Γ=Γ, Δt=Δt,
                                                                   window=window,
                                                                   n_cycles=long_n_cycles,
                                                                   p=p,
                                                                   ens_size=ens_obs_size,
                                                                   D=D, t0=0.0)

observations = observations[(end - n_cycles + 1):end, :]
true_states = true_states[(end - n_cycles + 1):end, :]
ensembles = ensembles[(end - n_cycles + 1):end, :, :]

analyses_filtered = Filtering.filtering_cycles(; ensemble=ensemble, model=model, h=h,
                                               observations=observations,
                                               integrator=integrator,
                                               filter_method=filter_method,
                                               ens_size=ens_size, Δt=Δt,
                                               window=window, n_cycles=n_cycles,
                                               model_size=model_size, Γ=Γ,
                                               assimilate_obs=true,
                                               calc_score=false,
                                               max_cycle=max_cycle, t0=t0)

ensemble = x0 .+ 0.25 * randn(D, ens_size)
analyses_unfiltered = Filtering.filtering_cycles(; ensemble=ensemble, model=model, h=h,
                                                 observations=observations,
                                                 integrator=integrator,
                                                 filter_method=filter_method,
                                                 ens_size=ens_size, Δt=Δt,
                                                 window=window, n_cycles=n_cycles,
                                                 model_size=model_size, Γ=Γ,
                                                 assimilate_obs=false,
                                                 t0=t0)

dists = [Metrics.wasserstein(analyses_unfiltered[i, :, :], ensembles[i, :, :], ens_size,
                             ens_size) for i in 1:n_cycles]
dists_da = [Metrics.wasserstein(analyses_filtered[i, :, :], ensembles[i, :, :], ens_size,
                                ens_size) for i in 1:n_cycles]