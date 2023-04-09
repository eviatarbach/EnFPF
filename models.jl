module Models

function System(func_generic, params)
   func = (t, u)->func_generic(t, u, params)
   return func
end

function lorenz63_func(t, u, p)
   du = similar(u)
   du[1] = p["σ"]*(u[2]-u[1])
   du[2] = u[1]*(p["ρ"]-u[3]) - u[2]
   du[3] = u[1]*u[2] - p["β"]*u[3]

   return copy(du)
end

lorenz63 = System(lorenz63_func, Dict("σ" => 10, "β" => 8/3, "ρ" => 28))

function lorenz96_func(t, u, p)
   N = p["N"]

   du = similar(u)

   for i=1:N
      du[i] = (u[mod(i+1, 1:N)] - u[mod(i-2, 1:N)])*u[mod(i-1, 1:N)] - u[i] + p["F"]
   end

   return copy(du)
end

lorenz96 = System(lorenz96_func, Dict("F" => 8, "N" => 40))

function lorenz96_twoscale_func(t, u, p)
   N = p["N"]
   n = p["n"]

   dx = zeros(N)
   dy = zeros(n, N)

   u = reshape(u, n + 1, N)
   x = u[1, :]
   y = u[2:end, :]

   for i=1:N
      dx[i] = (x[mod(i+1, 1:N)] - x[mod(i-2, 1:N)])*x[mod(i-1, 1:N)] - x[i] + p["F"] - p["h"]*p["c"]/p["b"]*sum(y[:, i])

      for j=1:n
         if j == n
           jp1 = 1
           jp2 = 2
           jm1 = n - 1
           ip1 = mod(i + 1, 1:N)
           ip2 = mod(i + 1, 1:N)
           im1 = i
         elseif j == n - 1
           jp1 = n
           jp2 = 1
           jm1 = n - 2
           ip1 = i
           ip2 = mod(i + 1, 1:N)
           im1 = i
         elseif  j == 1
           jp1 = 2
           jp2 = 3
           jm1 = n
           ip1 = i
           ip2 = i
           im1 = mod(i - 1, 1:N)
         else
           jp1 = j + 1
           jp2 = j + 2
           jm1 = j - 1
           ip1 = ip2 = im1 = i
         end
         dy[j, i] = p["c"]*p["b"]*y[jp1, ip1]*(y[jm1, im1] - y[jp2, ip2]) - p["c"]*y[j, i] + p["h"]*p["c"]/p["b"]*x[i]
      end
   end

   du = vec([dx dy']')

   return du
end

lorenz96_twoscale = System(lorenz96_twoscale_func, Dict("F" => 8, "b" => 10, "c" => 10, "h" => 1.0, "N" => 40, "n" => 10))

end
