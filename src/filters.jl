module Filters

export ensrf

using Statistics
using LinearAlgebra

using Distributions

function enfpf(; E::AbstractMatrix{float_type}, R::AbstractMatrix{float_type}, H,
                 y::AbstractVector{float_type},
                 calc_score="gaussian") where {float_type<:AbstractFloat}
    D, m = size(E)

    x_m = mean(E, dims=2)
    
    HE = hcat([H(E[:, i]) for i=1:m]...)
    y_m = mean(HE, dims=2)

    x_m = mean(E, dims=2)
    X = E .- x_m

    K = PH*inv(HPH + R)
    E .+= K*(y + rand(MvNormal(R)) - y_m)

    if calc_score == "gaussian"
        E += -K*R*K'*inv((X*X')/(m-1))*X
    elseif calc_score[1] == "kernel"
        score_estimator = calc_score[2]
        score_estimator.fit(Matrix{Float32}(E)')
        score = score_estimator.compute_gradients(Matrix{Float32}(E)').numpy()'
        E += K*R*K'*score
    end

    return E
end

end
