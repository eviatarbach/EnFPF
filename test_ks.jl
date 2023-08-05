using Statistics
using LinearAlgebra
using Random

using Distributions

import EnFPF.DA
import EnFPF.DA_methods
import EnFPF.Models
import EnFPF.Integrators
import EnFPF.Metrics

Random.seed!(13)

D = 128
model = Models.KuramotoSivashinsky

p = 64*2
ens_size = 100
ens_obs_size = 100
model_size = D
integrator = Integrators.ks
da_method = DA_methods.ensrf

x0 = randn(D)
t0 = 0.0
Δt = 0.25
outfreq = 1
transient = 2000
x = integrator(model, x0, t0, transient*outfreq*Δt, Δt, inplace=false)
max_cycle = 30

H(v) = vcat([v[1:64].^i for i=1:2]...)

n_cycles = 200

assimilate_obs = true
save_P_hist = false

leads = 1
x0 = x[end, :]

window = 8

ensemble = x0 .+ 0.1*randn(D, ens_size)
ensemble_obs = x[1000, :] .+ 0.1*randn(D, ens_obs_size)

R = diagm(0=>1e-16*ones(p))

true_states, ensembles, observations, covariance = DA.make_observations(ensemble=ensemble_obs, model_true=model, H_true=H, integrator=integrator,
                                           R=1e-16*R, Δt=Δt, window=window, n_cycles=n_cycles, outfreq=outfreq,
                                           p=p, ens_size=ens_obs_size, D=D)

R = cov(observations[100:end, :], dims=1) + 1e-8*I(p)
R = diagm(0=>diag(R))/5
obs_err_dist = MvNormal(R)

observations = (mean(observations[200:end, :], dims=1)[:] .+ rand(obs_err_dist, n_cycles))'

da_info = DA.da_cycles(ensemble=ensemble, model=model, H=H, observations=observations, integrator=integrator,
                       da_method=da_method, ens_size=ens_size, Δt=Δt,
                       window=window, n_cycles=n_cycles, outfreq=outfreq,
                       model_size=model_size, R=R,
                       assimilate_obs=assimilate_obs,
                       leads=leads, save_P_hist=save_P_hist, calc_score="gaussian", max_cycle=max_cycle)

filtered = hcat([mean([H(da_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles]...)'

ensemble = x0 .+ 0.1*randn(D, ens_size)
noda_info = DA.da_cycles(ensemble=ensemble, model=model, H=H, observations=observations, integrator=integrator,
                       da_method=da_method, ens_size=ens_size, Δt=Δt,
                       window=window, n_cycles=n_cycles, outfreq=outfreq,
                       model_size=model_size, R=R,
                       assimilate_obs=false,
                       leads=leads, save_P_hist=save_P_hist)

nfiltered = hcat([mean([H(noda_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles]...)'

invariant = reshape(permutedims(ensembles[end:end, 1:64, :], [2,3,1]), 64, :)
dists = mean([[Metrics.wasserstein(noda_info.analyses[i, j:j, :], invariant[j:j, :], ens_size, ens_size) for i=1:n_cycles] for j=1:64])
dists_da = mean([[Metrics.wasserstein(da_info.analyses[i, j:j, :], invariant[j:j, :], ens_size, ens_size) for i=1:n_cycles] for j=1:64])