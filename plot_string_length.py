import numpy as np, matplotlib.pyplot as plt
import prototype_string_detection, cosmology

strings = np.loadtxt("strings.dat")
step, string_id, x, y, z = strings.T

steps = []
lengths = []

for step_used in np.unique(step):
    for i in np.unique(string_id[step == step_used]):
        length = np.sum((step == step_used) & (string_id == i)) * 3/2 * prototype_string_detection.dx
        steps.append(step_used)
        lengths.append(length)

tau = cosmology.tau_start + np.array(steps) * cosmology.dtau
log = cosmology.tau_to_log(tau)

t = cosmology.tau_to_t(tau)
zeta = np.array(lengths) / cosmology.L**3 * t**2

# I think this is not right

plt.scatter(log, zeta)
plt.xlabel(r"$\log(m_r / H)")
plt.ylabel(r"$\zeta = l t^2 / L^3$")
plt.show()
