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

Random.seed!(1)

moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

D = 3
model = Models.lorenz63

p = 2
ens_size = 10
ens_obs_size = 100
model_size = D
integrator = Integrators.rk4
da_method = DA_methods.etkf

x0 = randn(D)
t0 = 0.0
Δt = 0.05
outfreq = 1
transient = 2000
x = integrator(model, x0, t0, transient*outfreq*Δt, Δt, inplace=false)

H(v) = diag(v*v')[1:2]#[v[1] > quantile(x[:, 1], 0.5), v[2] > quantile(x[:, 2], 0.5),
        #v[3] > quantile(x[:, 3], 0.5)]
H2(v) = diag(v*v')[3:3]

n_cycles = 500

vec_cov = reshape(var([vec(x[i, :]*x[i, :]') for i=1:transient])/ens_size, 3, 3)
R = Symmetric(diagm(0.01*diag(vec_cov)[1:2]))#Symmetric(diagm([0.05, 0.05, 0.05])
ens_err = Symmetric(diagm(0.25*ones(D)))
assimilate_obs = true
save_P_hist = true

leads = 1
x0 = x[end, :]

window = 40

ensemble = x0 .+ 0.25*randn(D, ens_size)
ensemble_obs = x[1000, :] .+ 0.25*randn(D, ens_obs_size)

#observations = hcat([cov(x) .+ rand(MvNormal(R), ens_size) for i=1:n_cycles]...)'

true_states, true_hidden, observations, covariance = DA.make_observations(ensemble=ensemble_obs, model_true=model, H_true=H, integrator=integrator,
                                           R=R, Δt=Δt, window=window, n_cycles=n_cycles, outfreq=outfreq,
                                           p=p, ens_size=ens_obs_size, H_hidden=H2, p_hidden=1)

da_info = DA.da_cycles(ensemble=ensemble, model=model, H=H, observations=observations, integrator=integrator,
                       da_method=da_method, localization=nothing, ens_size=ens_size, Δt=Δt,
                       window=window, n_cycles=n_cycles, outfreq=outfreq,
                       model_size=model_size, R=R,# + covariance,
                       assimilate_obs=assimilate_obs,
                       leads=leads, save_P_hist=save_P_hist)

#info = DA.compute_stats(da_info=da_info, trues=true_states)

filtered = hcat([mean([H(da_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles]...)'
# sqrt.(mean((filtered - true_states).^2, dims=1))
# mean([std([H(da_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles])