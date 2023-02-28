import os.path, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, fftfreq
from numba import jit
import cosmology, string_detection, load_data

# replaces scipy.fft.fftn for testing purposes
#import pyfftw
#def fftn(xs):
#    xs = xs.astype("complex")
#    out = np.empty_like(xs)
#    plan = pyfftw.FFTW(xs, out, axes=(0,1,2), direction="FFTW_FORWARD")
#    plan.execute()
#    return out

# get \dot{\theta} from \phi and \dot{\phi}
def compute_theta_dot(phi, phi_dot, a):
    d_theta_d_tau = (
        (phi_dot.imag * phi.real - phi.imag * phi_dot.real) /
        (phi.real**2 - phi.imag**2)
    )
    theta_dot = d_theta_d_tau / a
    return theta_dot

# 0 close to strings, 1 everyhere else
def compute_W(phi):
    N = phi.shape[0]
    strings = string_detection.is_string_at(phi)
    ps = np.array(list(zip(*np.where(strings))))
    d_min = int(np.ceil(np.sqrt(3) * 1))
    W = np.ones((N, N, N))
    #return W # NOTE: for debugging, this should give the same spectrum as without the ppse
    offsets = [(d1, d2, d3)
        for d1 in range(-d_min, d_min + 1)
            for d2 in range(-d_min, d_min + 1)
                for d3 in range(-d_min, d_min + 1)
                    if np.linalg.norm([d1, d2, d3]) <= d_min]
    for p in ps:
        for (d1, d2, d3) in offsets:
            W[(p[0] + d1) % N, (p[1] + d2) % N, (p[2] + d3) % N] = 0
    return W

# find all the grid points belonging to surface of the sphere with
# radius between min_k and max_k i.e. belonging to a certain bin in
# the histogram
@jit
def find_sphere(k_1d, k_abs_3d, min_k, max_k):
    out = []
    N = k_1d.size
    for i1 in range(N):
        for i2 in range(N):
            for i3 in range(N):
                if min_k <= k_abs_3d[i1, i2, i3] < max_k:
                    out.append((i1, i2, i3))
    return np.array(out)

# P_field(k) = k^2 / L^3 \int d \Omega / 4\pi 0.5 * | field(k) |^2
def compute_spectrum(field, L, spheres, bin_k, surface_element):
    integrant = np.abs(fftn(field))**2 # TODO: is k**2 right here or bin_?????
    spectrum = np.array([
            np.sum(integrant[spheres[i].T[0], spheres[i].T[1], spheres[i].T[2]])
            for i in range(len(bin_k))])
    spectrum *= bin_k**2 / L**3 / (4*np.pi) * 0.5 * surface_element
    return np.array(spectrum)

@jit
def substract_wave_numbers(f, idx1, idx2, N):
    # here we use k * 2pi
    # convert from index to wave number
    k, k_prime = f[idx1], f[idx2]
    # substract wave_numbers (mod N)
    k_diff = (k - k_prime + N//2) % N - N//2
    # converet from wave number to index
    idx_diff = k_diff + N if k_diff < 0 else k_diff
    return idx_diff

# sum |\tilde W(k - k')|^2 for given indicies of wavevectors
# belonging to the k or k' sphere respectifly
@jit
def sum_spheres(sphere1, sphere2, N, f, integrant):
    out = 0.0
    for idx1_1, idx1_2, idx1_3 in sphere1:
        for idx2_1, idx2_2, idx2_3  in sphere2:
            c = integrant[
                    substract_wave_numbers(f, idx1_1, idx2_1, N),
                    substract_wave_numbers(f, idx1_2, idx2_2, N),
                    substract_wave_numbers(f, idx1_3, idx2_3, N),
            ]
            out += c.real**2 + c.imag**2
    return out

def compute_spheres(N, nbins, bin_width, k_1d, k_abs_3d):
    # compute all the spheres
    spheres = []
    for i in range(nbins):
        min_k = i * bin_width
        max_k = min_k + bin_width
        spheres.append(find_sphere(k_1d, k_abs_3d, min_k, max_k))
    return spheres

# M = 1 / (L^3)^2 * \int d \Omega / 4\pi d \Omega' / 4\pi |W(\vec{k} - \vec{k}')|^2
def compute_M(N, nbins, W_fft, surface_element, L):
    f = np.round(fftfreq(N) * N).astype("int")
    # cache the sum for M as they take a lot of time to compute
    fname = os.path.join(os.path.dirname(__file__), "M.npy")
    if os.path.exists(fname):
        M = np.load(fname)
    else:
        M = np.empty((nbins, nbins))
        for i in range(nbins):
            for j in range(i, nbins):
                M[i, j] = sum_spheres(spheres[i], spheres[j], N, f, W_fft)
                M[j, i] = M[i, j]
                print(i, j, nbins)
        np.save(fname, M)

    # prefactors and integration measures
    for i in range(nbins):
        for j in range(nbins):
            M[i, j] *= surface_element[i] * surface_element[j]
    M /= (L**6 * (4*np.pi)**2)
    return M

def compute_M_inv(M, bin_width, bin_k):
    # definition of M^-1:
    # \int k'^2 dk' / 2\pi^2 M^{-1}(k, k') M(k', k'') = 2\pi^2/k^2 \delta(k - k'')
    # bin_width * sum_{k'} k'^2 / (2*np.pi^2) * M^-1(k,k') M(k',k'') = 2pi^2/k^2 delta(k - k'')
    # tilde M^-1 = bin_width / (2pi^2)^2 * k^2 k'^2 * M^-1(k, k')
    # M^-1(k, k') = (2pi^2)^2 / (bin_width * k^2 k'^2) * tilde M^-1(k, k')
    # sum_k' tilde M^-1(k, k')  M(k', k'') = delta(k, k'')
    M_inv = np.linalg.inv(M)
    M_inv *= (2*np.pi**2)**2 / (bin_width * bin_k[:, None]**2 * bin_k[None, :]**2)
    return M_inv

# load data from simulation and calculate cosmological quantities
data = load_data.OutputDir("run1_output")
N, phi, phi_dot, dx, L = data.N, data.final_field, data.final_field_dot, data.dx, data.L

a = cosmology.tau_to_a(data.tau_final)
dx_physical = dx * a
# TODO: physical or comoving coordinates? (I think its the physical)
k_1d = 2*np.pi * fftfreq(N, dx_physical) # wave numbers along one dimensions
k1, k2, k3 = np.meshgrid(k_1d, k_1d, k_1d) # 3d grid of wave numbers
k_abs_3d = np.sqrt(k1**2 + k2**2 + k3**2)

# histogram for functions of |k| e.g. the spectrum P(|k|)
k_max = (
    (N / 2 if N % 2 == 0 else (N - 1) / 2) *
    1 / (dx_physical * N) *
    2*np.pi
    # * np.sqrt(3)
) # assert np.isclose(k.max(), k_max)
nbins = 20
bin_width = k_max / nbins # k_min = 0
bin_start = np.arange(nbins) * bin_width
bin_end = bin_start + bin_width
bin_k = bin_start + bin_width/2

# prefactors for surface integration on the grid
vol = 4/3*np.pi * ((bin_start + bin_width)**3 - bin_start**3)
area = 4*np.pi * bin_k**2
Delta_k = k_1d[1] - k_1d[0] # spacing of the wave numbers
surface_element = Delta_k**3 / vol * area

W = compute_W(phi)
spheres = compute_spheres(N, nbins, bin_width, k_1d, k_abs_3d)
W_fft = fftn(W)
M = compute_M(N, nbins, W_fft, surface_element, L)
M_inv = compute_M_inv(M, bin_width, bin_k)

theta_dot = compute_theta_dot(phi, phi_dot, a)
theta_dot_tilde = theta_dot * W

# full P = spectrum ( \dot{\theta} )
P_full = compute_spectrum(theta_dot, L, spheres, bin_k, surface_element)
# \tilde P = spectrum ( \dot{\theta} * W )
P_tilde = compute_spectrum(theta_dot_tilde, L, spheres, bin_k, surface_element)
# \hat P = M_inv spectrum ( \dot{\theta} * W )
P_ppse = bin_k**2 / L**3 / (2*np.pi**2) * bin_width * M_inv @ P_tilde

#P_ppse *= bin_k**4 # TODO: what why ???????
#missing_prefactor = np.mean(P_tilde / P_ppse) # TODO: what is the correct prefactor
#P_ppse *= missing_prefactor

def plot():
    # plot all spectra computed
    plt.figure(layout="constrained")
    plt.step(bin_start, P_full, where="pre", label="full spectrum including strings")
    plt.step(bin_start, P_tilde, ls="--", where="pre", label="masked uncorrected spectrum")
    plt.step(bin_start, P_ppse, where="pre", label="PPSE of free axions")
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"log = {data.log_end:.2f}")
    plt.legend()
    plt.show()

py_field = theta_dot_tilde
py_fft = fftn(py_field)

theta_dot_c = np.loadtxt("run1_output/theta_dot.dat").reshape(N, N, N)
W_c = np.loadtxt("run1_output/W.dat").reshape(N, N, N)
c_field = np.loadtxt("run1_output/masked_theta_dot_out.dat").reshape(N,N,N)
c_fft = np.loadtxt("run1_output/masked_theta_dot_fft_out.dat", dtype="complex").reshape(N,N,N)

assert np.isclose(theta_dot.transpose(2,1,0) / theta_dot_c, 1).all()
assert np.all(W.transpose(2,1,0) == W_c)
# field = W * theta_dot
assert np.isclose(py_field.transpose(2,1,0) - c_field, 0).all()
assert np.isclose(py_fft.transpose(2,1,0) / c_fft, 0.5-0.5j).all()
