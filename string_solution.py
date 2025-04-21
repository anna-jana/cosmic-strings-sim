import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

def solution_for_small_rho(x, c): return c*x**5*(c**2 + 1/8)/24 - c*x**3/8 + c*x
def solution_for_large_rho(x): return 1 - 1/(2*x**2) - 3/(2*x**3)

def rhs(u, y):
    f, df = y
    return np.array([df, + df + f + np.exp(u)**2 * f * (f**2 - 1)])

def bc(left, right):
    return np.array([left[0], right[0] - 1.0])

u_min = np.log(1e-15) # u = log(rho)
u_max = np.log(10)
npoints = 200
us = np.linspace(u_min, u_max, npoints)
f_guess = np.vstack([us, np.ones(npoints)])
npoints = 500

sol = solve_bvp(rhs, bc, us, f_guess)
assert sol.success
rho = np.exp(sol.x)
f = sol.y[0, :]

# c should be the derivative at rho = 0 but this is difficult numerically
c = np.max(np.diff(f) / np.diff(rho))

plt.figure()
plt.plot(rho, 1 - f, label="numerical solution")
ylims = plt.ylim()
plt.plot(rho, 1 - solution_for_small_rho(rho, c), label=r"approx. for small $\rho$")
plt.plot(rho, 1 - solution_for_large_rho(rho), label=r"approx. for large $\rho$")
plt.ylim(ylims)
plt.xlabel(r"radial distance to string $\rho$")
plt.ylabel(r"dimensionless radial component $r / f_a$ of PQ field $\frac{r + f_a}{\sqrt{2}} e**{i \theta}$")
plt.legend()
plt.title("single string")
plt.savefig("string_solution.pdf")
plt.show()
