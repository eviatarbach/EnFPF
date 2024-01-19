module Filters

export enfpf

using Statistics
using LinearAlgebra

using Distributions

"""The ensemble Fokker—Planck filter (EnFPF)"""
function enfpf(; E::AbstractMatrix{float_type}, Γ::AbstractMatrix{float_type}, h,
               y::AbstractVector{float_type},
               calc_score=false) where {float_type<:AbstractFloat}
    D, m = size(E)

    hE = hcat([h(E[:, i]) for i in 1:m]...)

    x_m = mean(E; dims=2)
    y_m = mean(hE; dims=2)

    X = E .- x_m
    Y = hE .- y_m

    C_vh = (X * Y') / (m - 1)
    C_hh = (Y * Y') / (m - 1)

    K = C_vh * inv(C_hh + Γ)
    E .+= K * (y + rand(MvNormal(Γ)) - y_m)

    if calc_score == "gaussian"
        C_vv = (X * X') / (m - 1)
        E += -K * Γ * K' * inv(C_vv) * X
    elseif calc_score[1] == "kernel"
        score_estimator = calc_score[2]
        score_estimator.fit(Matrix{Float32}(E)')
        score = score_estimator.compute_gradients(Matrix{Float32}(E)').numpy()'
        E += K * Γ * K' * score
    end

    return E
end

"""A square-root formulation of the EnFPF"""
function senfpf(; E::AbstractMatrix{float_type}, Γ::AbstractMatrix{float_type}, h,
               y::AbstractVector{float_type},
               calc_score=false) where {float_type<:AbstractFloat}
    D, m = size(E)

    hE = hcat([h(E[:, i]) for i in 1:m]...)

    v_m = mean(E; dims=2)
    y_m = mean(hE; dims=2)

    V = (E .- v_m)/sqrt(m-1)
    Y = (hE .- y_m)/sqrt(m-1)

    Γ_inv = inv(Γ)

    W = Γ_inv - Γ_inv*Y*(I + Y'*Γ_inv*Y)^-1*Y'*Γ_inv
    K = V*Y'*W

    E .+= K * (y + rand(MvNormal(Γ)) - y_m)

    return E
end

end
