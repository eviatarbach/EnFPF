using Statistics
using LinearAlgebra
using Random

using Distributions

import EnFPF.Filtering
import EnFPF.Filters
import EnFPF.Models
import EnFPF.Integrators
import EnFPF.Metrics

Random.seed!(13)

D = 128
model = Models.KuramotoSivashinsky

p = 64 * 2
ens_size = 100
ens_obs_size = 100
model_size = D
integrator = Integrators.ks
filter_method = Filters.enfpf

x0 = randn(D)
t0 = 0.0
Δt = 0.25
transient = 2000
x = integrator(model, x0, t0, transient * Δt, Δt; inplace=false)
max_cycle = 30

h(v) = vcat([v[1:64] .^ i for i in 1:2]...)

n_cycles = 200

x0 = x[end, :]

window = 8

ensemble = x0 .+ 0.1 * randn(D, ens_size)
ensemble_obs = x[1000, :] .+ 0.1 * randn(D, ens_obs_size)

true_states, ensembles, observations = Filtering.make_observations(; ensemble=ensemble_obs,
                                                                   model_true=model, h=h,
                                                                   integrator=integrator,
                                                                   Γ=nothing, Δt=Δt,
                                                                   window=window,
                                                                   n_cycles=n_cycles,
                                                                   p=p,
                                                                   ens_size=ens_obs_size,
                                                                   D=D)

Γ = cov(observations[100:end, :]; dims=1)
Γ = diagm(0 => diag(Γ)) / 5
obs_err_dist = MvNormal(Γ)

observations = (mean(observations[200:end, :]; dims=1)[:] .+ rand(obs_err_dist, n_cycles))'

analyses_filtered = Filtering.filtering_cycles(; ensemble=ensemble, model=model, h=h,
                                               observations=observations,
                                               integrator=integrator,
                                               filter_method=filter_method,
                                               ens_size=ens_size, Δt=Δt,
                                               window=window, n_cycles=n_cycles,
                                               model_size=model_size, Γ=Γ,
                                               assimilate_obs=true,
                                               calc_score="gaussian", max_cycle=max_cycle)

ensemble = x0 .+ 0.1 * randn(D, ens_size)
analyses_unfiltered = Filtering.filtering_cycles(; ensemble=ensemble, model=model, h=h,
                                                 observations=observations,
                                                 integrator=integrator,
                                                 filter_method=filter_method,
                                                 ens_size=ens_size, Δt=Δt,
                                                 window=window, n_cycles=n_cycles,
                                                 model_size=model_size, Γ=Γ,
                                                 assimilate_obs=false)

invariant = reshape(permutedims(ensembles[end:end, 1:64, :], [2, 3, 1]), 64, :)
dists = mean([[Metrics.wasserstein(analyses_unfiltered[i, j:j, :], invariant[j:j, :],
                                   ens_size, ens_size) for i in 1:n_cycles] for j in 1:64])
dists_da = mean([[Metrics.wasserstein(analyses_filtered[i, j:j, :], invariant[j:j, :],
                                      ens_size, ens_size) for i in 1:n_cycles] for j in 1:64])