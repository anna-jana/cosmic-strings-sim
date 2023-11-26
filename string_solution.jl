using DifferentialEquations
using PyPlot

solution_for_small_rho(x, c) = c*x^5*(c^2 + 1/8)/24 - c*x^3/8 + c*x
solution_for_large_rho(x) = 1 - 1/(2*x^2) - 3/(2*x^3)

function rhs!(dy, y, p, u)
    f, df = y
    dy[1] = df
    dy[2] = + df + f + exp(u)^2 * f * (f^2 - 1)
end

function bc_left!(residual, y, p)
    residual[1] = y[1]
end

function bc_right!(residual, y, p)
    residual[1] = y[1] - one(y[1])
end

u_min = log(1e-5) # u = log(rho)
u_max = log(10)
initial_guess = [0.0, 1.0]
npoints = 500
problem = TwoPointBVProblem(rhs!, (bc_left!, bc_right!),
                            initial_guess, (u_min, u_max);
                            bcresid_prototype = (zeros(1), zeros(1)))
solution = solve(problem, MIRK4(), dt = u_max / npoints, reltol=1e-10, abstol=1e-10)
rho = exp.(solution.t)
f = solution[1, :]

# c should be the derivative at rho = 0 but this is difficult numerically
base = 2500
c = (f[2 + base] - f[1 + base]) / (rho[2 + base] - rho[1 + base])

figure()
plot(rho, @. 1 - f; label="numerical solution")
ylims = ylim()
plot(rho, @. 1 - solution_for_small_rho(rho, c); label=raw"approx. for small $\rho$")
plot(rho, @. 1 - solution_for_large_rho(rho); label=raw"approx. for large $\rho$")
ylim(ylims)
xlabel(raw"radial distance to string $\rho$")
ylabel(raw"dimensionless radial component $r / f_a$ of PQ field $\frac{r + f_a}{\sqrt{2}} e^{i \theta}$")
legend()
title("single string")
savefig("string_solution.pdf")
show()
