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

D = 3
model = Models.lorenz63

H(v) = diag(v*v')
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
R = Symmetric(diagm(10000*ones(3)))
ens_err = Symmetric(diagm(0.25*ones(D)))
assimilate_obs = false
save_P_hist = true

leads = 1
x0 = x[end, :]

n_cycles = 2000*leads
ρ = 1e-2

window = 1

ensemble = x0 .+ 0.25*randn(D, ens_size)
ensemble_obs = x0 .+ 0.25*randn(D, ens_obs_size)

#observations = hcat([cov(x) .+ rand(MvNormal(R), ens_size) for i=1:n_cycles]...)'

true_states, observations = DA.make_observations(ensemble=ensemble_obs, model_true=model, H_true=H, integrator=integrator,
                                           R=R, Δt=Δt, window=window, n_cycles=n_cycles, outfreq=outfreq,
                                           p=3, ens_size=ens_obs_size)

da_info = DA.da_cycles(ensemble=ensemble, model=model, H=H, observations=observations, integrator=integrator,
                       da_method=da_method, localization=nothing, ens_size=ens_size, Δt=Δt,
                       window=window, n_cycles=n_cycles, outfreq=outfreq,
                       model_size=model_size, R=R, ρ=ρ,
                       assimilate_obs=assimilate_obs,
                       leads=leads, save_P_hist=save_P_hist)

#info = DA.compute_stats(da_info=da_info, trues=true_states)