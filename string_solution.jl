using DifferentialEquations
using PyPlot

function rhs!(dy, y, p, u)
    r, dr = y
    dy[1] = dr
    dy[2] = + dr + r + exp(u)^2 * r * (r^2 - 1)
end

function bc_left!(residual, y, p)
    residual[1] = y[1] - one(y[1])
end

function bc_right!(residual, y, p)
    residual[1] = y[1]
end

u_min = log(1e-5) # u = log(r)
u_max = log(5)
initial_guess = [0.0, 1.0]
npoints = 500
problem = TwoPointBVProblem(rhs!, (bc_left!, bc_right!),
                            initial_guess, (u_min, u_max);
                            bcresid_prototype = (zeros(1), zeros(1)))
solution = solve(problem, MIRK4(), dt = u_max / npoints)
rho = exp.(solution.t)
r = solution[1, :]

figure()
semilogx(rho, r)
xlabel("radial distance to string")
ylabel(raw"radial component of PQ field $r(\rho)$")
title("single string")
savefig("string_solution.pdf")
show()
