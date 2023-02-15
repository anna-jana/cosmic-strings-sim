import numpy as np, matplotlib.pyplot as plt
import cosmology, load_data

# 2/3 factor from moore appendix A.2

steps = []
lengths = []
for step_used in np.unique(load_data.string_step):
    for i in np.unique(load_data.string_id[load_data.string_step == step_used]):
        length = np.sum((load_data.string_step == step_used) & (load_data.string_id == i)) * 2/3 * load_data.dx
        steps.append(step_used)
        lengths.append(length)

tau = load_data.tau_start + np.array(steps) * load_data.dtau
t = cosmology.tau_to_t(tau)
log = cosmology.tau_to_log(tau)

plt.subplot(2,1,1)
plt.scatter(log, lengths)
plt.xlabel(r"$\log(m_r / H)$")
plt.ylabel("lengths")

steps = []
lengths = []
for step_used in np.unique(load_data.string_step):
    length = np.sum(load_data.string_step == step_used) * 2/3 * load_data.dx
    steps.append(step_used)
    lengths.append(length)

tau = load_data.tau_start + np.array(steps) * load_data.dtau
t = cosmology.tau_to_t(tau)
log = cosmology.tau_to_log(tau)
a = cosmology.tau_to_a(tau)

zeta = np.array(lengths) * a / (a * load_data.L)**3 * t**2

plt.subplot(2,1,2)
plt.plot(log, zeta)
plt.xlabel(r"$\log(m_r / H)$")
plt.ylabel(r"$\zeta = l t^2 / L^3$")

plt.tight_layout()

plt.show()
