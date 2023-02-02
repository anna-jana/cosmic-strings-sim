import numpy as np, matplotlib.pyplot as plt
from scipy.fft import fftn, fftfreq
import cosmology
from cosmology import N, dx, L

phi = np.reshape(np.loadtxt("final_field.dat", dtype="complex"), (N, N, N))
phi_dot = np.reshape(np.loadtxt("final_field_dot.dat",  dtype="complex"), (N, N, N))

a = cosmology.tau_to_a(cosmology.tau_end)

d_theta_d_tau = (
    (phi_dot.imag * phi.real - phi.imag * phi_dot.real) /
    (phi.real**2 - phi.imag**2)
)
theta_dot = d_theta_d_tau / a
theta_dot_fft = fftn(theta_dot).ravel()

k_1d = 2*np.pi * fftfreq(N, dx) # TODO: physical or comoving coordinates?
k1, k2, k3 = np.meshgrid(k_1d, k_1d, k_1d)
k = np.sqrt(k1**2 + k2**2 + k3**2).ravel()

k_max = (N / 2) / (dx*N) if N % 2 == 0 else ((N - 1) / 2) / (dx*N)
k_max *= 2*np.pi*np.sqrt(3)
k_min = 0.0
nbins = 20
bin_width = (k_max - k_min) / nbins
bin_index = np.floor(k / bin_width).astype("int")

integrant = k**2 * np.abs(theta_dot_fft)**2 / (2*np.pi*L)**3
spectrum = [np.sum(integrant[bin_index == i]) for i in range(0, nbins)]

plt.figure(layout="constrained")
plt.step(np.arange(nbins) * bin_width, spectrum, where="pre")
plt.xlabel("k")
plt.ylabel("P(k)")
#plt.xscale("log")
plt.yscale("log")
plt.show()

# TODO: this spectrum is contaminated by initial axions and strings!!!!
# TODO: implement string screening
