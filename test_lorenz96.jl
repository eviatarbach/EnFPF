using Statistics
using LinearAlgebra
using Random

using Distributions

import EnFPF.DA
import EnFPF.DA_methods
import EnFPF.Models
import EnFPF.Integrators
import EnFPF.Metrics

Random.seed!(4)

D = 40
model = Models.lorenz96

ens_size = 100
ens_obs_size = 100
model_size = D
integrator = Integrators.rk4
da_method = DA_methods.ensrf

n_moments = 2
p = D*n_moments

x0 = randn(D)
t0 = 0.0
Δt = 0.05
outfreq = 1
transient = 2000
x = integrator(model, x0, t0, transient*outfreq*Δt, Δt, inplace=false)
max_cycle = 40

H(v) = vcat([v.^i for i=1:n_moments]...)

n_cycles = 300

assimilate_obs = true
save_P_hist = true

leads = 1
x0 = x[end, :]

window = 1

ensemble = x0 .+ 0.2*randn(D, ens_size)
ensemble_obs = x[1000, :] .+ 0.2*randn(D, ens_obs_size)

R = diagm(0=>1e-16*ones(p))

true_states, ensembles, observations, covariance = DA.make_observations(ensemble=ensemble_obs, model_true=model, H_true=H, integrator=integrator,
                                           R=1e-16*R, Δt=Δt, window=window, n_cycles=n_cycles, outfreq=outfreq,
                                           p=p, ens_size=ens_obs_size, D=D)

R = cov(observations[200:end, :], dims=1) + 1e-8*I(p)
R = diagm(0=>diag(R))/5
obs_err_dist = MvNormal(R)

observations = (mean(observations[200:end, :], dims=1)[:] .+ rand(obs_err_dist, n_cycles))'

da_info = DA.da_cycles(ensemble=ensemble, model=model, H=H, observations=observations, integrator=integrator,
                       da_method=da_method, ens_size=ens_size, Δt=Δt,
                       window=window, n_cycles=n_cycles, outfreq=outfreq,
                       model_size=model_size, R=R,
                       assimilate_obs=assimilate_obs,
                       leads=leads, save_P_hist=save_P_hist, calc_score=false, max_cycle=max_cycle)

filtered = hcat([mean([H(da_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles]...)'

ensemble = x0 .+ 0.2*randn(D, ens_size)
noda_info = DA.da_cycles(ensemble=ensemble, model=model, H=H, observations=observations, integrator=integrator,
                       da_method=da_method, ens_size=ens_size, Δt=Δt,
                       window=window, n_cycles=n_cycles, outfreq=outfreq,
                       model_size=model_size, R=R,
                       assimilate_obs=false,
                       leads=leads, save_P_hist=save_P_hist)

nfiltered = hcat([mean([H(noda_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles]...)'

invariant = reshape(permutedims(ensembles[end:end, :, :], [2,3,1]), 40, :)
dists = mean([[Metrics.wasserstein(noda_info.analyses[i, j:j, :], invariant[j:j, :], ens_size, ens_size) for i=1:300] for j=1:40])
dists_da = mean([[Metrics.wasserstein(da_info.analyses[i, j:j, :], invariant[j:j, :], ens_size, ens_size) for i=1:300] for j=1:40])