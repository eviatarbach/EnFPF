module DA_methods

export etkf, ensrf

using Statistics
using LinearAlgebra
using Distributions

"""
Ensemble transform Kalman filter (ETKF)
"""
function etkf(; E::AbstractMatrix{float_type}, R::AbstractMatrix{float_type},
                R_inv::AbstractMatrix{float_type},
                inflation::float_type=1.0, H,
                y::AbstractVector{float_type}, localization=nothing) where {float_type<:AbstractFloat}
    D, m = size(E)

    x_m = mean(E, dims=2)
    X = (E .- x_m)/sqrt(m - 1)

    X = sqrt(inflation)*X

    y_m = H(x_m)
    Y = (hcat([H(E[:, i]) for i=1:m]...) .- y_m)/sqrt(m - 1)
    Ω = inv(Symmetric(I + Y'*R_inv*Y))
    w = Ω*Y'*R_inv*(y - y_m)

    E = x_m .+ X*(w .+ sqrt(m - 1)*sqrt(Ω))

    return E
end

function ensrf(; E::AbstractMatrix{float_type}, R::AbstractMatrix{float_type},
                 R_inv::AbstractMatrix{float_type},
                 inflation::float_type=1.0, H,
                 y::AbstractVector{float_type},
                 localization=nothing) where {float_type<:AbstractFloat}
    D, m = size(E)

    x_m = mean(E, dims=2)
    A = E .- x_m

    if localization === nothing
        P = inflation*A*A'/(m - 1)
    else
        P = inflation*localization.*(A*A')/(m - 1)
    end

    K = P*H'*inv(H*P*H' + R)
    x_m .+= K*(y - H*x_m)

    E = x_m .+ real((I + P*H'*R_inv*H)^(-1/2))*A

    return E
end

end
