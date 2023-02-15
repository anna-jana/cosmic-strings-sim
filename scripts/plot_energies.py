import numpy as np, matplotlib.pyplot as plt
import cosmology, load_data

tau = load_data.tau_start + load_data.energy_step * load_data.dtau
log = cosmology.tau_to_log(tau)

plt.figure()

plt.plot(log, load_data.axion_kinetic, color="tab:blue", ls="-", label="axion, kinetic")
plt.plot(log, load_data.axion_gradient, color="tab:blue", ls="--", label="axion, gradient")
plt.plot(log, load_data.axion_total, color="tab:blue", ls="-", lw=2, label="axion")
plt.plot(log, load_data.radial_kinetic, color="tab:orange", ls="-", label="radial, kinetic")
plt.plot(log, load_data.radial_gradient, color="tab:orange", ls="--", label="radial, gradient")
plt.plot(log, load_data.radial_potential, color="tab:orange", ls=":", label="radial, potential")
plt.plot(log, load_data.radial_total, color="tab:orange", ls="-", lw=2, label="radial")
plt.plot(log, load_data.interaction, color="tab:green", ls="-", label="interaction axion and radial = strings")
plt.plot(log, load_data.total, color="black", ls="-", lw=2, label="total energy density")

plt.yscale("log")
#plt.xlabel(r"conformal time, $\tau$")
plt.xlabel(r"$log(m_r / H)$")
ax = plt.gca()
#ax.invert_xaxis()
plt.ylabel(r"averaged energy density $f_a^2 m_r^2$\n")
plt.legend()
plt.show()
