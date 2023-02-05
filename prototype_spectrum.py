import numpy as np, matplotlib.pyplot as plt
from scipy.fft import fftn, fftfreq
from numba import jit
import os.path
import cosmology, prototype_string_detection

from cosmology import N, dx, L

def compute_theta_dot(phi, phi_dot, a):
    d_theta_d_tau = (
        (phi_dot.imag * phi.real - phi.imag * phi_dot.real) /
        (phi.real**2 - phi.imag**2)
    )
    theta_dot = d_theta_d_tau / a
    return theta_dot

def compute_W(phi):
    N = phi.shape[0]
    strings = prototype_string_detection.is_string_at(phi)
    ps = np.array(list(zip(*np.where(strings))))
    d_min = int(np.ceil(np.sqrt(3) * 3))
    W = np.ones((N, N, N))
    offsets = [(d1, d2, d3)
        for d1 in range(-d_min, d_min + 1)
            for d2 in range(-d_min, d_min + 1)
                for d3 in range(-d_min, d_min + 1)
                    if np.linalg.norm([d1, d2, d3]) <= d_min]
    for p in ps:
        for (d1, d2, d3) in offsets:
            W[(p[0] + d1) % N, (p[1] + d2) % N, (p[2] + d3) % N] = 0
    return W

def compute_spectrum(field, L, k, nbins, bin_index):
    field_fft = fftn(field).ravel()
    integrant = k**2 * np.abs(field_fft)**2 / (2*np.pi*L)**3
    spectrum = [np.sum(integrant[bin_index == i]) for i in range(nbins)]
    return np.array(spectrum)

phi = np.reshape(np.loadtxt("final_field.dat", dtype="complex"), (N, N, N))
phi_dot = np.reshape(np.loadtxt("final_field_dot.dat",  dtype="complex"), (N, N, N))
a = cosmology.tau_to_a(cosmology.tau_end)

# dealing with momenta (independent of string screening method)
dx_physical = dx * a
# TODO: physical or comoving coordinates? (I think its the physical)
k_1d = 2*np.pi * fftfreq(N, dx_physical)
k1, k2, k3 = np.meshgrid(k_1d, k_1d, k_1d)
k = np.sqrt(k1**2 + k2**2 + k3**2).ravel()
k_max = N / 2 if N % 2 == 0 else (N - 1) / 2
k_max /= dx_physical * N
k_max *= 2*np.pi*np.sqrt(3)
k_min = 0.0
nbins = 50
bin_width = (k_max - k_min) / nbins
bin_index = np.floor(k / bin_width).astype("int")

theta_dot = compute_theta_dot(phi, phi_dot, a)
W = compute_W(phi)
theta_dot_tilde = theta_dot * W
P_tilde = compute_spectrum(theta_dot_tilde, L, k, nbins, bin_index)
W_fft = fftn(W).ravel()

@jit
def sum_spheres(idxs1, idxs2, N, integrant):
    out = 0.0
    for idx1 in idxs1:
        for idx2 in idxs2:
            idx = (idx1 - idx2) % N
            c = integrant[idx]
            out += c.real**2 + c.imag**2
    return out

@jit
def find_sphere(k, min_k, max_k):
    out = []
    for i in range(k.size):
        if min_k <= k[i] < max_k:
            out.append(i)
    return np.array(out)

fname = "M.npy"
if os.path.exists(fname):
    M = np.load(fname)
else:
    M = np.empty((nbins, nbins))
    idxs = []
    for i in range(nbins):
        print(i)
        min_k = i * bin_width
        max_k = min_k + bin_width
        idxs.append(find_sphere(k, min_k, max_k))
    for i in range(nbins):
        for j in range(i, nbins):
            M[i, j] = sum_spheres(idxs[i], idxs[j], N, W_fft)
            M[j, i] = M[i, j]
            print(i, j, nbins)
    M /= (L**6 * (4*np.pi)**2)
    np.save(fname, M)

M_inv = np.linalg.inv(M)

bins = np.arange(nbins) * bin_width
bin_k = bins + bin_width
ppse_spectrum = bin_k**2 / L**3 / (2*np.pi**2) * M_inv @ P_tilde

naive_spectrum = compute_spectrum(theta_dot, L, k, nbins, bin_index)

# plot
plt.figure(layout="constrained")
plt.step(bins, naive_spectrum, where="pre", label="naive (no-screening)")
plt.step(bins, ppse_spectrum, where="pre", label="PPSE")
plt.xlabel("k")
plt.ylabel("P(k)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

