module Models

using FFTW, Statistics

function System(func_generic, params)
   func = (t, u) -> func_generic(t, u, params)
   return func
end

function lorenz63_func(t, u, p)
   du = similar(u)
   du[1] = p["σ"] * (u[2] - u[1])
   du[2] = u[1] * (p["ρ"] - u[3]) - u[2]
   du[3] = u[1] * u[2] - p["β"] * u[3]

   return copy(du)
end

lorenz63 = System(lorenz63_func, Dict("σ" => 10, "β" => 8 / 3, "ρ" => 28))

function lorenz63_na_func(t, u, p)
   """
   From Daron, J. D., & Stainforth, D. A. (2015). On quantifying the climate of the nonautonomous Lorenz-63 model.
   Chaos, 25(4), 043103. https://doi.org/10.1063/1.4916789
   """
   du = similar(u)
   Ψ = p["A"] * (1 / 3 * sin(2 * pi * p["f"] * t) + 1 / 3 * sin(sqrt(3) * p["f"] * t) + 1 / 3 * sin(sqrt(17) * p["f"] * t))
   du[1] = p["σ"] * (u[2] - u[1])
   du[2] = u[1] * ((p["ρ"] + Ψ) - u[3]) - u[2]
   du[3] = u[1] * u[2] - p["β"] * u[3]

   return copy(du)
end

lorenz63_na = System(lorenz63_na_func, Dict("σ" => 10, "β" => 8 / 3, "ρ" => 28, "f" => 1, "A" => 3))

function lorenz96_func(t, u, p)
   N = p["N"]

   du = similar(u)

   for i = 1:N
      du[i] = (u[mod(i + 1, 1:N)] - u[mod(i - 2, 1:N)]) * u[mod(i - 1, 1:N)] - u[i] + p["F"]
   end

   return copy(du)
end

lorenz96 = System(lorenz96_func, Dict("F" => 8, "N" => 40))

Q = 64
N = 22
h = 0.25
M = 16
#Precompute various ETDRK4 scalar quantities
k = 2π / N * (0.0:Q-1)
L = (k .^ 2 - k .^ 4) # Fourier Multiples
E = exp.(h * L)
E2 = exp.(h * L / 2)
M = 16 # No. of points for complex means
r = exp.(im * π * ((1:M) .- 0.5) / M)
LR = h * L * ones(M)' + ones(Q) * r'
QQ = h * real.(mean((exp.(LR / 2) .- 1) ./ LR, dims=2))[:]
f1 = h * real(mean((-4 .- LR + exp.(LR) .* (4 .- 3 * LR + LR .^ 2)) ./ LR .^ 3, dims=2))[:]
f2 = h * real(mean((2 .+ LR + exp.(LR) .* (-2 .+ LR)) ./ LR .^ 3, dims=2))[:]
f3 = h * real(mean((-4 .- 3 * LR - LR .^ 2 + exp.(LR) .* (4 .- LR)) ./ LR .^ 3, dims=2))[:]
g = -0.5im * k

function KuramotoSivashinsky(u, tmax; h=h)
   """
   From https://github.com/JuliaDynamics/TimeseriesPrediction.jl/blob/master/test/ks_solver.jl,
   based on exponential differencing fourth-order Runge–Kutta from
   Kassam, A.-K., & Trefethen, L. N. (2005). Fourth-Order Time-Stepping for Stiff PDEs.
   SIAM Journal on Scientific Computing, 26(4), 1214–1233. https://doi.org/10.1137/S1064827502410633
   """
   v = zeros(ComplexF64, Q)
   T = plan_fft(v)
   Ti = plan_ifft(v)
   T! = plan_fft!(v)
   Ti! = plan_ifft!(v)

   a = Complex.(zeros(Q))
   b = Complex.(zeros(Q))
   c = Complex.(zeros(Q))
   Nv = Complex.(zeros(Q))
   Na = Complex.(zeros(Q))
   Nb = Complex.(zeros(Q))
   Nc = Complex.(zeros(Q))

   u_real = u[1:Q]
   u_imag = u[Q+1:end]
   u = complex.(u_real, u_imag)
   nmax = round(Int, tmax / h)
   #v = fft(u)
   tt = 0.0:h:tmax
   uu = zeros(2 * Q, length(tt) - 1)

   for n = 1:nmax
      v = fft(u)
      Nv .= g .* (T * real(Ti * v) .^ 2) #.+ cc
      @. a = E2 * v + QQ * Nv
      Na .= g .* (T! * real(Ti! * a) .^ 2) #.+ cc
      @. b = E2 * v + QQ * Na
      Nb .= g .* (T! * real(Ti! * b) .^ 2) #.+ cc
      @. c = E2 * a + QQ * (2Nb - Nv)
      Nc .= g .* (T! * real(Ti! * c) .^ 2) #.+ cc
      @. v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3

      u = ifft(v)
      uu[1:Q, n] = real(u)
      uu[Q+1:end, n] = imag(u)
   end
   return uu
end

end
