import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

Lambda = 400 # [MeV]
M_pl = 2.435e18 * 1e3 # [MeV]
g_star = 104.98 # [1]
alpha = 1.68e-7 # [1]
alpha0 = 1.46e-3
n = 6.68 # [1]
zero_T = np.sqrt(alpha0) * Lambda**2

def axion_mass_times_f_a(T):
    power_law = np.sqrt(alpha) * (T / Lambda)**(-n/2) * Lambda**2
    return np.minimum(zero_T, power_law)

def hubble_at_temperature(T):
    return np.sqrt(g_star / 90) * np.pi / M_pl * T**2

def solve(f_a):
    def goal(log_T, f_a):
        T = np.exp(log_T)
        return np.log(axion_mass_times_f_a(T) / f_a / hubble_at_temperature(T))
    ans = root(goal, np.log(1e10 * Lambda), args=(f_a,))
    assert ans.success
    return np.exp(ans.x[0])

f_a_list = np.geomspace(1e8, 1e12) * 1e3 # [MeV]
T_osc = np.array([solve(f_a) for f_a in f_a_list]) # [MeV]
H_osc = hubble_at_temperature(T_osc) # [MeV]
H_osc_analytic = (
        (np.sqrt(g_star / 90) * np.pi / M_pl)**(n/(n + 4)) *
        alpha**(2/(n + 4)) *
        f_a_list**(- 4/(n + 4)) *
        Lambda**2
)
log = np.log(f_a_list / H_osc)
log_analytic = np.log(f_a_list / H_osc_analytic)

plt.figure()
plt.semilogx(f_a_list, log, label="numerical")
plt.semilogx(f_a_list, log_analytic, ls="--", label="analytical")
plt.xlabel("f_a / MeV")
plt.ylabel("log(f_a / H_osc)")
plt.legend()
plt.show()
