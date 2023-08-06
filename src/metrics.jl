module Metrics

export wasserstein

using PyCall

const ot = PyNULL()

function __init__()
    copy!(ot, pyimport("ot"))
end

function wasserstein(ens1, ens2, ens_size1, ens_size2)
    M = ot.dist(ens1', ens2', metric="euclidean")
    return ot.emd2(ones(ens_size1) / ens_size1, ones(ens_size2) / ens_size2, M)
end

end