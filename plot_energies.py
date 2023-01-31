import numpy as np, matplotlib.pyplot as plt

step, axion_kinetic, axion_gradient, axion_total, radial_kinetic, \
    radial_gradient, radial_potential, radial_total, interaction, total = np.loadtxt("energies.dat").T

dt = 1e-2
t = step * dt

plt.figure()

plt.plot(t, axion_kinetic, label="axion, kinetic")
plt.plot(t, axion_gradient, label="axion, gradient")
plt.plot(t, axion_total, label="axion")
plt.plot(t, radial_kinetic, label="radial, kinetic")
plt.plot(t, radial_gradient, label="radial, gradient")
plt.plot(t, radial_potential, label="radial, potential")
plt.plot(t, radial_total, label="radial")
plt.plot(t, interaction, label="interaction axion and radial = strings")
plt.plot(t, total, label="total energie density")

plt.yscale("log")
plt.xlabel(r"conformal time, $\tau$")
plt.ylabel(r"averaged energy density $f_a^2 m_r^2$\n")
plt.legend()
plt.show()
