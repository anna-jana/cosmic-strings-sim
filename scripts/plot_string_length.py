import numpy as np, matplotlib.pyplot as plt
import cosmology

strings = np.loadtxt("strings.dat")
step, string_id, x, y, z = strings.T

# 2/3 factor from moore appendix A.2

steps = []
lengths = []
for step_used in np.unique(step):
    for i in np.unique(string_id[step == step_used]):
        length = np.sum((step == step_used) & (string_id == i)) * 2/3 * cosmology.dx
        steps.append(step_used)
        lengths.append(length)

tau = cosmology.tau_start + np.array(steps) * cosmology.dtau
t = cosmology.tau_to_t(tau)
log = cosmology.tau_to_log(tau)

plt.subplot(2,1,1)
plt.scatter(log, lengths)
plt.xlabel(r"$\log(m_r / H)$")
plt.ylabel("lengths")

steps = []
lengths = []
for step_used in np.unique(step):
    length = np.sum(step == step_used) * 2/3 * cosmology.dx
    steps.append(step_used)
    lengths.append(length)

tau = cosmology.tau_start + np.array(steps) * cosmology.dtau
t = cosmology.tau_to_t(tau)
log = cosmology.tau_to_log(tau)
a = cosmology.tau_to_a(tau)

zeta = np.array(lengths) * a / (a * cosmology.L)**3 * t**2

plt.subplot(2,1,2)
plt.plot(log, zeta)
plt.xlabel(r"$\log(m_r / H)$")
plt.ylabel(r"$\zeta = l t^2 / L^3$")

plt.tight_layout()

plt.show()
