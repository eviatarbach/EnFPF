using Statistics
using LinearAlgebra
using Random

using Distributions

include("da.jl")
import .DA

include("da_methods.jl")
import .DA_methods

include("models.jl")
import .Models

include("integrators.jl")
import .Integrators

Random.seed!(10)

moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

D = 128
model = Models.KuramotoSivashinsky

p = 64*2
ens_size = 100
ens_obs_size = 100
model_size = D
integrator = Integrators.ks
da_method = DA_methods.ensrf
localization = nothing#DA.gaspari_cohn_localization(1, D, cyclic=true)

x0 = randn(D)
t0 = 0.0
Δt = 0.25
outfreq = 1
transient = 2000
x = integrator(model, x0, t0, transient*outfreq*Δt, Δt, inplace=false)
max_cycle = 30

#H(v) = diag(v*v')#[v[1] > quantile(x[:, 1], 0.5), v[2] > quantile(x[:, 2], 0.5),
        #v[3] > quantile(x[:, 3], 0.5)]
H(v) = vcat([v[1:64].^i for i=1:2]...)

n_cycles = 1000

#vec_cov = reshape(var([vec(x[i, :]*x[i, :]') for i=1:transient])/ens_size, D, D)
#R = diagm(0=>ones(p))#Symmetric(diagm(0.1*diag(vec_cov)))#Symmetric(diagm([0.05, 0.05, 0.05])
ens_err = Symmetric(diagm(0.25*ones(D)))
assimilate_obs = true
save_P_hist = false

leads = 1
x0 = x[end, :]

window = 4

ensemble = x0 .+ 0.25*randn(D, ens_size)
ensemble_obs = x[1000, :] .+ 0.25*randn(D, ens_obs_size)

R = diagm(0=>1e-16*ones(p))

#observations = hcat([cov(x) .+ rand(MvNormal(R), ens_size) for i=1:n_cycles]...)'

true_states, ensembles, observations, covariance = DA.make_observations(ensemble=ensemble_obs, model_true=model, H_true=H, integrator=integrator,
                                           R=1e-16*R, Δt=Δt, window=window, n_cycles=n_cycles, outfreq=outfreq,
                                           p=p, ens_size=ens_obs_size, D=D)

R = cov(observations[200:end, :], dims=1) + 1e-8*I(p)
R = diagm(0=>diag(R))
obs_err_dist = MvNormal(R)
m = mean(observations[200:end, 1:64], dims=1)[:]
v = mean(observations[200:end, 64+1:end], dims=1)[:]

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

ensemble = x0 .+ 0.25*randn(D, ens_size)
noda_info = DA.da_cycles(ensemble=ensemble, model=model, H=H, observations=observations, integrator=integrator,
                       da_method=da_method, localization=localization, ens_size=ens_size, Δt=Δt,
                       window=window, n_cycles=n_cycles, outfreq=outfreq,
                       model_size=model_size, R=R,
                       assimilate_obs=false,
                       leads=leads, save_P_hist=save_P_hist)

nfiltered = hcat([mean([H(noda_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles]...)'
# sqrt.(mean((filtered - true_states).^2, dims=1))
# mean([std([H(da_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles])