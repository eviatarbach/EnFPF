using Statistics
using LinearAlgebra
using Random

using Distributions

import EnFPF.Filtering
import EnFPF.Filters
import EnFPF.Models
import EnFPF.Integrators

Random.seed!(10)

D = 3
model = Models.lorenz63

ens_size = 10
ens_obs_size = 100
model_size = D
integrator = Integrators.rk4
filter_method = Filters.enfpf

x0 = randn(D)
t0 = 0.0
Δt = 0.05
transient = 2000
x = integrator(model, x0, t0, transient * Δt, Δt; inplace=false)

n_cycles = 1500

n_moments = 2
p = 6
h(v) = vcat([v .^ i for i in 1:n_moments]...)

ens_err = Symmetric(diagm(0.25 * ones(D)))

x0 = x[end, :]

window = 4
max_cycle = nothing

ensemble = x0 .+ 0.25 * randn(D, ens_size)
ensemble_obs = x[1000, :] .+ 0.25 * randn(D, ens_obs_size)

true_states, ensembles, observations = Filtering.make_observations(; ensemble=ensemble_obs,
                                                                   model_true=model,
                                                                   h=h,
                                                                   integrator=integrator,
                                                                   Γ=nothing,
                                                                   Δt=Δt,
                                                                   window=window,
                                                                   n_cycles=n_cycles,
                                                                   p=p,
                                                                   ens_size=ens_obs_size,
                                                                   D=D)

Γ = cov(observations[500:end, :]; dims=1) / 5
obs_err_dist = MvNormal(Γ)

observations = observations .+ rand(obs_err_dist, n_cycles)'

analyses_filtered = Filtering.filtering_cycles(; ensemble=ensemble,
                                               model=model,
                                               h=h,
                                               observations=observations,
                                               integrator=integrator,
                                               filter_method=filter_method,
                                               ens_size=ens_size,
                                               Δt=Δt,
                                               window=window,
                                               n_cycles=n_cycles,
                                               model_size=model_size,
                                               Γ=Γ,
                                               assimilate_obs=true,
                                               calc_score=false,
                                               max_cycle=max_cycle)

filtered = hcat([mean([h(analyses_filtered[i, :, j]) for j in 1:ens_size])
                 for i in 1:n_cycles]...)'

ensemble = x0 .+ 0.25 * randn(D, ens_size)

analyses_unfiltered = Filtering.filtering_cycles(; ensemble=ensemble,
                                                 model=model,
                                                 h=h,
                                                 observations=observations,
                                                 integrator=integrator,
                                                 filter_method=filter_method,
                                                 ens_size=ens_size,
                                                 Δt=Δt,
                                                 window=window,
                                                 n_cycles=n_cycles,
                                                 model_size=model_size,
                                                 Γ=Γ,
                                                 assimilate_obs=false)

nfiltered = hcat([mean([h(analyses_unfiltered[i, :, j]) for j in 1:ens_size])
                  for i in 1:n_cycles]...)'