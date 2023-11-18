using DifferentialEquations
using PyPlot

function rhs!(dy, y, p, u)
    f, df = y
    dy[1] = df
    dy[2] = + df + f + exp(u)^2 * f * (f^2 - 1)
end

function bc_left!(residual, y, p)
    residual[1] = y[1] - one(y[1])
end

function bc_right!(residual, y, p)
    residual[1] = y[1]
end

u_min = log(1e-5) # u = log(rho)
u_max = log(5)
initial_guess = [0.0, 1.0]
npoints = 500
problem = TwoPointBVProblem(rhs!, (bc_left!, bc_right!),
                            initial_guess, (u_min, u_max);
                            bcresid_prototype = (zeros(1), zeros(1)))
solution = solve(problem, MIRK4(), dt = u_max / npoints)
rho = exp.(solution.t)
f = solution[1, :]
r_over_f_a = @. 1 - f

figure()
semilogx(rho, r_over_f_a)
xlabel(raw"radial distance to string $\rho$")
ylabel(raw"dimensionless radial component $r / f_a$ of PQ field $\frac{r + f_a}{\sqrt{2}} e^{i \theta}$")
title("single string")
savefig("string_solution.pdf")
show()
