module Metrics

export wasserstein

using PythonCall

const ot = PythonCall.pynew()

function __init__()
    return PythonCall.pycopy!(ot, pyimport("ot"))
end

function wasserstein(ens1, ens2, ens_size1, ens_size2)
    M = ot.dist(Py(ens1').to_numpy(), Py(ens2').to_numpy(); metric="euclidean")
	return pyconvert(Float64, ot.emd2(Py(ones(ens_size1) / ens_size1).to_numpy(),
							          Py(ones(ens_size2) / ens_size2).to_numpy(), M))
end

end
