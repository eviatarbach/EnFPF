using Statistics
using LinearAlgebra
using Random

using Distributions
using BlockArrays

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

localization_P = DA.gaspari_cohn_localization(2, D, cyclic=true)
localization_HPH = Array(mortar(reshape([localization_P, zeros(D, D), zeros(D, D), localization_P], 2, 2)))
localization_PH = Array(mortar(reshape([localization_P, localization_P], 1, 2)))
localization = nothing#[localization_P, localization_HPH, localization_PH]

x0 = randn(D)
t0 = 0.0
Δt = 0.05
outfreq = 1
transient = 2000
x = integrator(model, x0, t0, transient*outfreq*Δt, Δt, inplace=false)
max_cycle = 40#Inf

#H(v) = diag(v*v')#[v[1] > quantile(x[:, 1], 0.5), v[2] > quantile(x[:, 2], 0.5),
        #v[3] > quantile(x[:, 3], 0.5)]
H(v) = vcat([v.^i for i=1:n_moments]...)

n_cycles = 300

#vec_cov = reshape(var([vec(x[i, :]*x[i, :]') for i=1:transient])/ens_size, D, D)
#R = diagm(0=>ones(p))#Symmetric(diagm(0.1*diag(vec_cov)))#Symmetric(diagm([0.05, 0.05, 0.05])
assimilate_obs = true
save_P_hist = true

leads = 1
x0 = x[end, :]

window = 1

ensemble = x0 .+ 0.2*randn(D, ens_size)
ensemble_obs = x[1000, :] .+ 0.2*randn(D, ens_obs_size)

R = diagm(0=>1e-16*ones(p))

#observations = hcat([cov(x) .+ rand(MvNormal(R), ens_size) for i=1:n_cycles]...)'

true_states, ensembles, observations, covariance = DA.make_observations(ensemble=ensemble_obs, model_true=model, H_true=H, integrator=integrator,
                                           R=1e-16*R, Δt=Δt, window=window, n_cycles=n_cycles, outfreq=outfreq,
                                           p=p, ens_size=ens_obs_size, D=D)

R = cov(observations[200:end, :], dims=1) + 1e-8*I(p)
R = diagm(0=>diag(R))/5
obs_err_dist = MvNormal(R)
#m = mean(observations[200:end, 1:20], dims=1)[:]
#v = mean(observations[200:end, 41:80], dims=1)[:]

observations = (mean(observations[200:end, :], dims=1)[:] .+ rand(obs_err_dist, n_cycles))'

#R = diagm(0=>var(observations[200:end, :], dims=1)[:])

#observations[:] .= repeat(mean(observations[200:end, :], dims=1)[:], 500)
#observations = repeat(mean(observations[200:end, :], dims=1), outer=500)

da_info = DA.da_cycles(ensemble=ensemble, model=model, H=H, observations=observations, integrator=integrator,
                       da_method=da_method, localization=localization, ens_size=ens_size, Δt=Δt,
                       window=window, n_cycles=n_cycles, outfreq=outfreq,
                       model_size=model_size, R=R,
                       assimilate_obs=assimilate_obs,
                       leads=leads, save_P_hist=save_P_hist, calc_score=false, max_cycle=max_cycle)

#info = DA.compute_stats(da_info=da_info, trues=true_states)

filtered = hcat([mean([H(da_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles]...)'

ensemble = x0 .+ 0.2*randn(D, ens_size)
noda_info = DA.da_cycles(ensemble=ensemble, model=model, H=H, observations=observations, integrator=integrator,
                       da_method=da_method, localization=localization, ens_size=ens_size, Δt=Δt,
                       window=window, n_cycles=n_cycles, outfreq=outfreq,
                       model_size=model_size, R=R,
                       assimilate_obs=false,
                       leads=leads, save_P_hist=save_P_hist)

nfiltered = hcat([mean([H(noda_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles]...)'
# sqrt.(mean((filtered - true_states).^2, dims=1))
# mean([std([H(da_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles])

invariant = reshape(permutedims(ensembles[end:end, :, :], [2,3,1]), 40, :)
dists = mean([[Metrics.wasserstein(noda_info.analyses[i, j:j, :], invariant[j:j, :], ens_size, ens_size) for i=1:300] for j=1:40])
dists_da = mean([[Metrics.wasserstein(da_info.analyses[i, j:j, :], invariant[j:j, :], ens_size, ens_size) for i=1:300] for j=1:40])