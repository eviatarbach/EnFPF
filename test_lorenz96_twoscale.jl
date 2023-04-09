using Statistics
using LinearAlgebra
using Random

using Distributions
using Zygote

include("da.jl")
import .DA

include("da_methods.jl")
import .DA_methods

include("models.jl")
import .Models

include("integrators.jl")
import .Integrators

Random.seed!(11)

moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

D = 40
D2 = 440
model_true = Models.lorenz96_twoscale
model = Models.lorenz96

indices = reshape(1:440, 11, :)
first_layer_indices = indices[1, :]
second_layer_indices = indices[2:end, :]

p = 40
ens_size = 10
ens_obs_size = 10
model_size = D
integrator = Integrators.rk4
da_method = DA_methods.ensrf
localization = DA.gaspari_cohn_localization(1, D, cyclic=true)

x0 = randn(D2)
t0 = 0.0
Δt = 0.005
outfreq = 1
transient = 2000
x = integrator(model_true, x0, t0, transient*outfreq*Δt, Δt, inplace=false)

H(v) = diag(v*v')
H2(v) = diag(v*v')[first_layer_indices]
H_linear(v) = jacobian(H, v)[1]

n_cycles = 1000

vec_cov = reshape(var([vec(x[i, first_layer_indices]*x[i, first_layer_indices]') for i=1:transient])/ens_size, D, D)
R = Symmetric(diagm(0.1*diag(vec_cov)))
ens_err = Symmetric(diagm(0.25*ones(D)))
assimilate_obs = true
save_P_hist = true

leads = 1
x0 = x[end, :]

window = 10

ensemble = x0[first_layer_indices] .+ 0.25*randn(D, ens_size)
ensemble_obs = x[1000, :] .+ 0.25*randn(D2, ens_obs_size)

true_states, observations, covariance = DA.make_observations(ensemble=ensemble_obs, model_true=model_true, H_true=H2, integrator=integrator,
                                           R=R, Δt=Δt, window=window, n_cycles=n_cycles, outfreq=outfreq,
                                           p=p, ens_size=ens_obs_size)

da_info = DA.da_cycles(ensemble=ensemble, model=model, H=H, H_linear=H_linear, observations=observations, integrator=integrator,
                       da_method=da_method, localization=localization, ens_size=ens_size, Δt=Δt,
                       window=window, n_cycles=n_cycles, outfreq=outfreq,
                       model_size=model_size, R=R,
                       assimilate_obs=assimilate_obs,
                       leads=leads, save_P_hist=save_P_hist, calc_score=true)

filtered = hcat([mean([H(da_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles]...)'

ensemble = x0[first_layer_indices] .+ 0.25*randn(D, ens_size)
noda_info = DA.da_cycles(ensemble=ensemble, model=model, H=H, H_linear=H_linear, observations=observations, integrator=integrator,
                       da_method=da_method, localization=localization, ens_size=ens_size, Δt=Δt,
                       window=window, n_cycles=n_cycles, outfreq=outfreq,
                       model_size=model_size, R=R,
                       assimilate_obs=false,
                       leads=leads, save_P_hist=save_P_hist)

nfiltered = hcat([mean([H(noda_info.analyses[i, :, j]) for j=1:ens_size]) for i=1:n_cycles]...)'