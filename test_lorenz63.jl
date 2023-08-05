using Statistics
using LinearAlgebra
using Random

using Distributions

import EnFPF.DA
import EnFPF.DA_methods
import EnFPF.Models
import EnFPF.Integrators
import EnFPF.Metrics

Random.seed!(10)

D = 3
model = Models.lorenz63

ens_size = 100
ens_obs_size = 100
model_size = D
integrator = Integrators.rk4
da_method = DA_methods.ensrf

x0 = randn(D)
t0 = 0.0
Δt = 0.05
outfreq = 1
transient = 2000
x = integrator(model, x0, t0, transient*outfreq*Δt, Δt, inplace=false)

n_cycles = 1500

mask = zeros(Bool, 3, 3)
for i=1:3
    for j=1:i
        mask[i, j] = true
    end
end

#H(vd) = [vd; (vd*vd')[mask]]

n_moments = 2
p = 6
#p = 3*n_moments
H(vd) = vcat([vd[1:3].^i for i=1:n_moments]...)

#vec_cov = reshape(var([vec(x[i, :]*x[i, :]') for i=1:transient])/ens_size, 3, 3)
R = diagm(0=>1e-16*ones(p))#Symmetric(diagm(0.1*diag(vec_cov)))#Symmetric(diagm([0.05, 0.05, 0.05])
ens_err = Symmetric(diagm(0.25*ones(D)))
assimilate_obs = true
save_P_hist = true

leads = 1
x0 = x[end, :]

window = 4
inflation = 1.0
max_cycle = 30

ensemble = x0 .+ 0.25*randn(D, ens_size)
ensemble_da = copy(ensemble)
ensemble_obs = x[1000, :] .+ 0.25*randn(D, ens_obs_size)

#observations = hcat([cov(x) .+ rand(MvNormal(R), ens_size) for i=1:n_cycles]...)'

true_states, ensembles, observations, covariance = DA.make_observations(ensemble=ensemble_obs, model_true=model, H_true=H, integrator=integrator,
                                           R=R, Δt=Δt, window=window, n_cycles=n_cycles, outfreq=outfreq,
                                           p=p, ens_size=ens_obs_size, D=D)

R = cov(observations[500:end, :], dims=1)/5 + 1e-8*I(p)
obs_err_dist = MvNormal(R)
# m = mean(observations[200:end, 1:3], dims=1)[:]
# v = mean(observations[200:end, 4:6], dims=1)[:]
#s = mean(observations[200:end, 7:9], dims=1)[:]

observations = (mean(observations[500:end, :], dims=1)[:] .+ rand(obs_err_dist, n_cycles))'
#observations = observations .+ rand(obs_err_dist, n_cycles)'

# true_states, observations, covariance = DA.make_observations(ensemble=ensemble_obs, model_true=model, H_true=vd->[(vd).^2; vd], integrator=integrator,
#                                            R=R, Δt=Δt, window=window, n_cycles=n_cycles, outfreq=outfreq,
#                                            p=p, ens_size=ens_obs_size)

#observations = repeat([v;m]', outer=500)

da_info = DA.da_cycles(ensemble=ensemble, model=model, H=H, observations=observations, integrator=integrator,
                       da_method=da_method, localization=nothing, ens_size=ens_size, Δt=Δt,
                       window=window, n_cycles=n_cycles, outfreq=outfreq,
                       model_size=model_size, R=R,
                       assimilate_obs=assimilate_obs,
                       leads=leads, save_P_hist=save_P_hist, calc_score=false, inflation=inflation,
                       max_cycle=max_cycle)

#info = DA.compute_stats(da_info=da_info, trues=true_states)
filtered = hcat([mean([H(da_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles]...)'
# sqrt.(mean((filtered - true_states).^2, dims=1))
# mean([std([H(da_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles])

ensemble = x0 .+ 0.25*randn(D, ens_size)
ensemble_noda = copy(ensemble)

noda_info = DA.da_cycles(ensemble=ensemble, model=model, H=H, observations=observations, integrator=integrator,
                       da_method=da_method, localization=nothing, ens_size=ens_size, Δt=Δt,
                       window=window, n_cycles=n_cycles, outfreq=outfreq,
                       model_size=model_size, R=R,
                       assimilate_obs=false,
                       leads=leads, save_P_hist=save_P_hist)

nfiltered = hcat([mean([H(noda_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles]...)'

invariant = reshape(permutedims(ensembles[end-14:end, :, :], [2,3,1]), D, :)

dists = [Metrics.wasserstein(noda_info.analyses[i, :, :], invariant, ens_size, 1500) for i=1:300]
dists_da = [Metrics.wasserstein(da_info.analyses[i, :, :], invariant, ens_size, 1500) for i=1:300]

# function wasserstein(ens1, ens2)
#     M = ot.dist(ens1',ens2', metric="euclidean")
#     return ot.emd2(ones(100)/100,ones(1500)/1500,M)
# end