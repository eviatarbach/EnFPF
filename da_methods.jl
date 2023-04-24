module DA_methods

export etkf, ensrf

using Statistics
using LinearAlgebra

using Distributions
using PyCall
push!(pyimport("sys")."path", "kscore")
kscore = pyimport("kscore")
score_estimator = kscore.estimators.NuMethod(lam=0.1, kernel=kscore.kernels.CurlFreeIMQ())

"""
Ensemble transform Kalman filter (ETKF)
"""
function etkf(; E::AbstractMatrix{float_type}, R::AbstractMatrix{float_type},
                R_inv::AbstractMatrix{float_type},
                inflation::float_type=1.0, H,
                y::AbstractVector{float_type}, localization=nothing, calc_score=true, H_linear=nothing, Δt) where {float_type<:AbstractFloat}
    keep_size = true
    D, m = size(E)

    x_m = mean(E, dims=2)    
    X = (E .- x_m)/sqrt(m - 1)

    X = sqrt(inflation)*X
    X_old = X

    #HE = hcat([H2(E[:, i]) for i=1:m]...)
    #y_m = mean(HE, dims=2)

    y_m = H(x_m)
    Y = (hcat([H(E[:, i]) for i=1:m]...) .- y_m)/sqrt(m - 1)
    Ω = inv(Symmetric(I + Y'*R_inv*Y))
    w = Ω*Y'*R_inv*(y - y_m)

    if calc_score
        score_estimator.fit(Matrix{Float32}(E)')
        score = score_estimator.compute_gradients(Matrix{Float32}(E)').numpy()'
    end

    K = X*Y'*inv(Y*Y' + R)

    E = x_m .+ X*(w .+ sqrt(m - 1)*sqrt(Ω))

    if keep_size
        x_m = mean(E, dims=2)

        X = (E .- x_m)/sqrt(m - 1)

        #E = x_m .+ sqrt(tr(X_old*X_old')/tr(X*X'))*X*sqrt(m-1)
        E = x_m .+ X_old*sqrt(m-1)
        #println(tr(X_old*X_old')/tr(X*X'))
    end

    if calc_score
        E += Δt*K*R*K'*score
    end

    return E
end

function ensrf(; E::AbstractMatrix{float_type}, R::AbstractMatrix{float_type},
                 R_inv::AbstractMatrix{float_type},
                 inflation::float_type=1.0, H, H_linear,
                 y::AbstractVector{float_type},
                 localization=nothing, calc_score=true, Δt) where {float_type<:AbstractFloat}
    D, m = size(E)

    x_m = mean(E, dims=2)
    #H_l = H_linear(x_m)
    A = E .- x_m

    if localization === nothing
        P = inflation*A*A'/(m - 1)
    else
        P = inflation*localization.*(A*A')/(m - 1)
    end
    
    HE = hcat([H(E[:, i]) for i=1:m]...)
    y_m = H(x_m)#mean(HE, dims=2)
    Y = HE .- y_m

    x_m = mean(E, dims=2)
    X = E .- x_m
    PH = (X*Y')/(m-1)
    HPH = (Y*Y')/(m-1)

    K = PH*inv(HPH + R)
    x_m .+= K*(y - y_m)

    if calc_score
        score_estimator.fit(Matrix{Float32}(E)')
        score = score_estimator.compute_gradients(Matrix{Float32}(E)').numpy()'
    end

    #E = x_m .+ real((I + P*H_l'*R_inv*H_l)^(-1/2))*A
    E = x_m .+ A

    if calc_score
        E += Δt*K*R*K'*score
    end

    return E
end

function senkf(; E::AbstractMatrix{float_type}, R::AbstractMatrix{float_type},
                 R_inv::AbstractMatrix{float_type},
                 inflation::float_type=1.0, H,
                 y::AbstractVector{float_type}, localization=nothing) where {float_type<:AbstractFloat}
    D, m = size(E)
    err_dist = MvNormal(R)

    errs = rand(err_dist, m)
    errs .-= mean(errs, dims=2)
    y_ens = y .+ errs
    R_u = cov(errs')

    HE = hcat([H(E[:, i]) for i=1:m]...)
    y_m = mean(HE, dims=2)
    Y = HE .- y_m

    x_m = mean(E, dims=2)
    X = E .- x_m
    PH = (X*Y')/(m-1)
    HPH = (Y*Y')/(m-1)
    K = PH*inv(HPH + R_u)

    for i=1:m
        E[:, i] = E[:, i] + K*(y_ens[:, i] - H(E[:, i]))
    end

    return E
end

end
