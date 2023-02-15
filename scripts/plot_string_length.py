import sys
import numpy as np, matplotlib.pyplot as plt
import cosmology, load_data

data = load_data.OutputDir(sys.argv[1])

# 2/3 factor from moore appendix A.2

steps = []
lengths = []
for step_used in np.unique(data.string_step):
    for i in np.unique(data.string_id[data.string_step == step_used]):
        length = np.sum((data.string_step == step_used) & (data.string_id == i)) * 2/3 * data.dx
        steps.append(step_used)
        lengths.append(length)

tau = data.tau_start + np.array(steps) * data.dtau
t = cosmology.tau_to_t(tau)
log = cosmology.tau_to_log(tau)

plt.subplot(2,1,1)
plt.scatter(log, lengths)
plt.xlabel(r"$\log(m_r / H)$")
plt.ylabel("lengths")

steps = []
lengths = []
for step_used in np.unique(data.string_step):
    length = np.sum(data.string_step == step_used) * 2/3 * data.dx
    steps.append(step_used)
    lengths.append(length)

tau = data.tau_start + np.array(steps) * data.dtau
t = cosmology.tau_to_t(tau)
log = cosmology.tau_to_log(tau)
a = cosmology.tau_to_a(tau)

zeta = np.array(lengths) * a / (a * data.L)**3 * t**2

plt.subplot(2,1,2)
plt.plot(log, zeta)
plt.xlabel(r"$\log(m_r / H)$")
plt.ylabel(r"$\zeta = l t^2 / L^3$")

plt.tight_layout()

plt.show()
