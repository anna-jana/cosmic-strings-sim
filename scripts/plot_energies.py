import numpy as np, matplotlib.pyplot as plt
import cosmology

step, axion_kinetic, axion_gradient, axion_total, radial_kinetic, \
    radial_gradient, radial_potential, radial_total, interaction, total = np.loadtxt("energies.dat").T

tau = cosmology.tau_start + step * cosmology.dtau
log = cosmology.tau_to_log(tau)

plt.figure()

plt.plot(log, axion_kinetic, color="tab:blue", ls="-", label="axion, kinetic")
plt.plot(log, axion_gradient, color="tab:blue", ls="--", label="axion, gradient")
plt.plot(log, axion_total, color="tab:blue", ls="-", lw=2, label="axion")
plt.plot(log, radial_kinetic, color="tab:orange", ls="-", label="radial, kinetic")
plt.plot(log, radial_gradient, color="tab:orange", ls="--", label="radial, gradient")
plt.plot(log, radial_potential, color="tab:orange", ls=":", label="radial, potential")
plt.plot(log, radial_total, color="tab:orange", ls="-", lw=2, label="radial")
plt.plot(log, interaction, color="tab:green", ls="-", label="interaction axion and radial = strings")
plt.plot(log, total, color="black", ls="-", lw=2, label="total energy density")

plt.yscale("log")
#plt.xlabel(r"conformal time, $\tau$")
plt.xlabel(r"$log(m_r / H)$")
ax = plt.gca()
#ax.invert_xaxis()
plt.ylabel(r"averaged energy density $f_a^2 m_r^2$\n")
plt.legend()
plt.show()
