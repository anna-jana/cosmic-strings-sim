import os.path, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, fftfreq
from numba import jit
import cosmology, string_detection, load_data

plt.ion()

data = load_data.OutputDir("../run1_output")
N, phi, phi_dot, dx, L = data.N, data.final_field, data.final_field_dot, data.dx, L
a = cosmology.tau_to_a(data.tau_end)

def compute_theta_dot(phi, phi_dot, a):
    d_theta_d_tau = (
        (phi_dot.imag * phi.real - phi.imag * phi_dot.real) /
        (phi.real**2 - phi.imag**2)
    )
    theta_dot = d_theta_d_tau / a
    return theta_dot

def compute_W(phi):
    N = phi.shape[0]
    strings = string_detection.is_string_at(phi)
    ps = np.array(list(zip(*np.where(strings))))
    d_min = int(np.ceil(np.sqrt(3) * 1))
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


# dealing with momenta (independent of string screening method)
dx_physical = dx * a
# TODO: physical or comoving coordinates? (I think its the physical)
k_1d = 2*np.pi * fftfreq(N, dx_physical)
Delta_k = k_1d[1] - k_1d[0]
k1, k2, k3 = np.meshgrid(k_1d, k_1d, k_1d)
k_abs_3d = np.sqrt(k1**2 + k2**2 + k3**2)
k = k_abs_3d.ravel()
k_max = (N / 2 if N % 2 == 0 else (N - 1) / 2) / (dx_physical * N) * 2*np.pi*np.sqrt(3)
k_min = 0.0

nbins = 20
bin_width = (k_max - k_min) / nbins
bin_index = np.floor(k / bin_width).astype("int")
bins = np.arange(nbins) * bin_width
bin_k = bins + bin_width/2

vol = [4/3*np.pi * (((i + 1)*bin_width)**3 - (i*bin_width)**3) for i in range(nbins)]
area = [4*np.pi * (i*bin_width + bin_width/2)**2 for i in range(nbins)]

def compute_spectrum(field, L, k, nbins, bin_index):
    field_fft = fftn(field).ravel()
    integrant = np.abs(field_fft)**2 / (2*np.pi*L)**3 # TODO: is k**2 right here or bin_?????
    spectrum = [bin_k[i]**2 * np.sum(integrant[bin_index == i]) * Delta_k**3 / vol[i] * area[i]
            for i in range(nbins)]
    return np.array(spectrum)

theta_dot = compute_theta_dot(phi, phi_dot, a)
W = compute_W(phi)
theta_dot_tilde = theta_dot * W
P_tilde = compute_spectrum(theta_dot_tilde, L, k, nbins, bin_index)
W_fft = fftn(W)

@jit
def sum_spheres(idxs1, idxs2, N, f, integrant):
    out = 0.0
    for idx1_1, idx1_2, idx1_3 in idxs1:
        for idx2_1, idx2_2, idx2_3  in idxs2:
            # convert from index to momentum
            k_1, k_2, k_3 = f[idx1_1], f[idx1_2], f[idx1_3]
            k_prime_1, k_prime_2, k_prime_3 = f[idx2_1], f[idx2_2], f[idx2_3]
            # substract momenta (mod N)
            k_diff_1 = (k_1 - k_prime_1 + N//2) % N - N//2
            k_diff_2 = (k_2 - k_prime_2 + N//2) % N - N//2
            k_diff_3 = (k_3 - k_prime_3 + N//2) % N - N//2
            # converet from momentum to index
            if k_diff_1 < 0:
                k_diff_1 += N
            if k_diff_2 < 0:
                k_diff_2 += N
            if k_diff_3 < 0:
                k_diff_3 += N
           #     #if N % 2 == 0:
           #     #    idx = N//2 - 1 + N//2 + k_diff
           #     #else:
           #     #    idx = (N - 1) // 2 + (N - 1) // 2 + k_diff
           #     idx_1 = N + k_diff
           # else:
           #     idx = k_diff
            c = integrant[k_diff_1, k_diff_2, k_diff_3]
            out += c.real**2 + c.imag**2
    return out

@jit
def find_sphere(k_abs_3d, min_k, max_k):
    out = []
    N = k_1d.size
    for i1 in range(N):
        for i2 in range(N):
            for i3 in range(N):
                if min_k <= k_abs_3d[i1, i2, i3] < max_k:
                    out.append((i1, i2, i3))
    return np.array(out)

f = np.round(fftfreq(N) * N).astype("int")
idxs = []
for i in range(nbins):
    min_k = i * bin_width
    max_k = min_k + bin_width
    idxs.append(find_sphere(k_abs_3d, min_k, max_k))

fname = "M.npy"
if os.path.exists(fname):
    M = np.load(fname)
else:
    M = np.empty((nbins, nbins))
    for i in range(nbins):
        for j in range(i, nbins):
            M[i, j] = sum_spheres(idxs[i], idxs[j], N, f, W_fft)
            M[j, i] = M[i, j]
            print(i, j, nbins)
    np.save(fname, M)

for i in range(nbins):
    for j in range(nbins):
        M[i, j] *= Delta_k**3 * Delta_k**3 * area[i] / vol[i] * area[j] / vol[j]
M /= (L**6 * (4*np.pi)**2)

# bin_width * sum_{k'} k'^2 / (2*np.pi^2) * M^-1(k,k') M(k',k'') = 2pi^2/k^2 delta(k - k'')
# tilde M^-1 = bin_with / (2pi^2)^2 * k^2 k'^2 * M^-1(k, k')
# M^-1(k, k') = (2pi^2)^2 / (bin_with * k^2 k'^2) * tilde M^-1(k, k')
# sum_k' tilde M^-1(k, k')  M(k', k'') = delta(k, k'')
M_inv = np.linalg.inv(M)
for i in range(nbins):
    for j in range(nbins):
        M_inv[i, j] *= (2*np.pi**2)**2 / (bin_width * bin_k[i]**2 * bin_k[j]**2)

ppse_spectrum = bin_k**2 / L**3 / (2*np.pi**2) * bin_width * M_inv @ P_tilde

full_spectrum = compute_spectrum(theta_dot, L, k, nbins, bin_index)

# plot
plt.figure(layout="constrained")
plt.step(bins, full_spectrum, where="pre", label="full spectrum including strings")
plt.step(bins, ppse_spectrum, where="pre", label="PPSE of free axions")
plt.xlabel("k")
plt.ylabel("P(k)")
plt.xscale("log")
plt.yscale("log")
plt.legend()

