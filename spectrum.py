import numpy as np
from scipy.linalg import inv
from scipy.fft import rfftfreq
from mpi4py import MPI
from mpi4py_fft import PFFT, DistArray
import AxionStrings
import energy

################################# helper functions ##################################
def compute_theta_dot(a : float, psi : np.complex128, psi_dot : np.complex128):
    R = np.real(psi)
    I = np.imag(psi)
    R_dot = np.real(psi_dot)
    I_dot = np.imag(psi_dot)
    d_theta_d_tau = (I_dot * R - I * R_dot) / (R**2 - I**2)
    return d_theta_d_tau / a

def calc_k_max_grid(N, d):
    if N % 2 == 0:
        return 2 * np.pi * (N / 2) / (d*N)
    else:
        return 2 * np.pi * ((N - 1) / 2) / (d*N)

# k = 2pi * integer_k / (Delta x * N)
def single_index_to_integer_k(N:int, i:int):
    if N % 2 == 0:
        return i if i < N // 2 else i - N
    else:
        return i if i <= (N - 1) // 2 else i - N

def single_integer_k_to_index(N:int, integer_k:int) -> int:
    return integer_k if integer_k >= 0 else integer_k + N

def index_to_integer_k(N:int, i:(int, int, int)) -> (int, int, int):
    return (
        single_index_to_integer_k(N, i[0]),
        single_index_to_integer_k(N, i[1]),
        single_index_to_integer_k(N, i[2]),
    )

def integer_k_to_index(N:int, integer_k:(int, int, int)) -> (int, int, int):
    ix = single_integer_k_to_index(N, integer_k[1])
    if ix >= N // 2 + 1:
        ix -= N // 2 + 1
    return (
        ix,
        single_integer_k_to_index(N, integer_k[2]),
        single_integer_k_to_index(N, integer_k[3]),
    )

def min_integer_k(N):
    return -N // 2 if N % 2 == 0 else -(N - 1) // 2

def max_integer_k(N):
    return (N - 1) // 2

def single_check_integer_k(integer_k:int, min_l, max_l):
    return integer_k > max_l or integer_k < min_l

def check_integer_k(integer_k:(int, int, int), min_l, max_l) -> bool:
    return (single_check_integer_k(integer_k[0], min_l, max_l) and
           single_check_integer_k(integer_k[1], min_l, max_l) and
           single_check_integer_k(integer_k[2], min_l, max_l))

def distributed_fft(s: AxionStrings.State, p: AxionStrings.Parameter, array, direction):
    darray = DistArray((p.N, p.N, p.N), [0, 1, 2], buffer=array, comm=s.comm)
    plan = PFFT(s.comm, darray.global_shape, axes=(0, 1, 2), dtype=darray.dtype)
    match direction:
        case "forward":
            return plan.forward(darray)
        case "backward":
            return plan.backward(darray)
        case _:
            raise ValueError(f"invalid direction {direction}")

def substract_wave_numbers_lookup(W_fft, p, min_l, max_l, i1, i2):
    # go from 1-based to 0-based
    i1 = (i1[0] - 1, i1[1] - 1, i1[2] - 1)
    i2 = (i2[0] - 1, i2[1] - 1, i2[2] - 1)
    # convert to integers l such that k = 2pi * l / (Delta x * N)
    k1 = index_to_integer_k(p.N, i1)
    k2 = index_to_integer_k(p.N, i2)
    # do the substraction in k space
    Delta_k = (k1[0] - k2[0], k1[1] - k2[1], k1[2] - k2[2])
    if check_integer_k(Delta_k, min_l, max_l):
        # if the difference is outside of the k-range -> ignore
        return 0.0
    else:
        # convert back to index
        ix, iy, iz = integer_k_to_index(p.N, Delta_k)
        # convert back to 1-based and index into W_fft
        return  W_fft[ix + 1, iy + 1, iz + 1]

def power_spectrum_utils(p:AxionStrings.Parameter, a):
    # prepare histogram
    dx_physical = p.dx * a
    kmax = calc_k_max_grid(p.N, dx_physical)
    physical_ks = rfftfreq(p.N, dx_physical) * 2*np.pi

    Delta_k = 2*np.pi / (p.N * dx_physical)
    bin_width = kmax / p.nbins

    bin_ks = [i * bin_width + bin_width/2 for i in range(p.nbins)]

    def compute_surface_element(i):
        vol = 4.0/3.0 * np.pi * (((i + 1)*bin_width)**3 - (i*bin_width)**3)
        area = 4*np.pi * (i*bin_width + bin_width/2)**2
        return area / vol * Delta_k**3 / bin_ks[i]**2
    surface_element = [compute_surface_element(i) for i in range(p.nbins)]

    return physical_ks, bin_width, surface_element, bin_ks

def compute_integration_spheres(p:AxionStrings.Parameter, physical_ks, bin_width):
    spheres = [[] for _ in range(p.nbins)]
    for i in range(p.nbins):
        bin_k_min = i * bin_width
        bin_k_max = bin_k_min + bin_width
        for ix in range(p.N // 2 + 1):
            for iy in range(p.N):
                for iz in range(p.N):
                    k2 = physical_ks[ix]**2 + physical_ks[iy]**2 + physical_ks[iz]**2
                    if k2 >= bin_k_min**2 and k2 <= bin_k_max**2:
                        spheres[i].append(ix, iy, iz)
    return spheres

##################################### helper for ppse #######################################
def compute_M(p:AxionStrings.Parameter, W_fft, spheres, surface_element):
    # compute M
    # M = 1 / (L**3)**2 * \int d \Omega / 4\pi d \Omega' / 4\pi |W(\vec{k} - \vec{k}')|**2
    # NOTE: this is the computationally most expensive part
    M = np.zeros(p.nbins, p.nbins)
    f = p.L**6 * (4 * np.pi)**2
    min_l = min_integer_k(p.N)
    max_l = max_integer_k(p.N)
    for i in range(p.nbins):
        for j in range(p.nbins):
            print(f"{(i, j)} of {(p.nbins, p.nbins)}")
            # integrate spheres
            s = 0.0
            for idx1 in spheres[i]:
                for idx2 in spheres[j]:
                    s += np.abs(substract_wave_numbers_lookup(
                              W_fft, p, min_l, max_l, idx1, idx2))**2
            s *= surface_element[i] * surface_element[j] / f
            M[i, j] = M[j, i] = s
    return M

def get_string_mask(p:AxionStrings.Parameter, strings):
    W = np.ones((p.N, p.N, p.N))
    for string in strings:
        for point in string:
            for x_offset in range(-p.radius, p.radius + 1):
                for y_offset in range(-p.radius, p.radius + 1):
                      for z_offset in range(-p.radius, p.radius + 1):
                        r2 = x_offset**2 + y_offset**2 + z_offset**2
                        if r2 <= p.radius**2:
                            ix = int(np.floor(int, point[1] + 1.0)) % p.N
                            iy = int(np.floor(int, point[2] + 1.0)) % p.N
                            iz = int(np.floor(int, point[3] + 1.0)) % p.N
                            W[ix, iy, iz] = 0.0
    return W

################################### compute spectra ####################################
def compute_power_spectrum(s: AxionStrings.State, p: AxionStrings.Parameter, field, spheres, surface_element, bin_ks):
    field_fft = distributed_fft(s, p, field)

    # P_field(k) = k**2 / L**3 \int d \Omega / 4\pi 0.5 * | field(k) |**2
    spectrum = np.zeros(p.nbins)
    for i in range(p.nbins):
        for (ix, iy, iz) in spheres[i]:
            spectrum[i] += np.abs(field_fft[ix, iy, iz])**2
        spectrum[i] *= surface_element[i]
        spectrum[i] *= 1 / p.L**3 / (4 * np.pi)**2 * 0.5
    return spectrum

# compute PPSE (pseudo-power-spectrum-estimator) of the theta-dot field
# -> the spectrum of number denseties of axtion
def compute_spectrum_ppse(p : AxionStrings.Parameter, s : AxionStrings.State, strings : list[list[np.ndarray[float]]]):
    raise NotImplementedError("ppse not yet ready for MPI")

    a = AxionStrings.tau_to_a(s.tau)
    theta_dot = compute_theta_dot(a, s.psi, s.psi_dot)

    # compute W (string mask)
    W = get_string_mask(p, strings)

    # mask out the strings in theta dot
    theta_dot *= W

    physical_ks, bin_width, surface_element, bin_ks = power_spectrum_utils(p, a)
    spheres = compute_integration_spheres(p, physical_ks, bin_width)

    spectrum_uncorrected = compute_power_spectrum(s, p, theta_dot, spheres, surface_element, bin_ks)

    W_fft = distributed_fft(s, p, W)

    M = compute_M(p, W_fft, spheres, surface_element)

    M_inv = inv(M)

    # the definition of M_inv is not exactly matrix inverse but
    # is an integral (hence the bin_width) as well some integration measure factors
    for j in range(p.nbins):
        for i in range(p.nbins):
            M_inv[i, j] *= (2 * np.pi**2)**2 / (bin_width * bin_ks[i]**2 * bin_ks[j]**2)

    spectrum_corrected = M_inv @ spectrum_uncorrected

    # also applying M**-1 to P is not exactly matrix multiplication
    # hence again factors
    spectrum_corrected *= bin_ks**2 * bin_width / p.L**3 / (2*np.pi**2)

    return bin_ks, spectrum_corrected, spectrum_uncorrected

def compute_spectrum_autoscreen(p : AxionStrings.Parameter, s : AxionStrings.State):
    a = AxionStrings.tau_to_a(s.tau)
    local_psi = s.psi[1:-1, 1:-1, 1:-1]
    theta_dot = compute_theta_dot(a, local_psi, s.psi_dot[1:-1, 1:-1, 1:-1])
    r = energy.compute_radial_mode(local_psi, a)
    screened_theta_dot = (1 + r) * theta_dot

    physical_ks, bin_width, surface_element, bin_ks = power_spectrum_utils(p, a)

    # rfft of theta_dot
    screended_theta_dot_fft = distributed_fft(s, p, screened_theta_dot)

    # collect the parts of the integration for our node
    my_range = screended_theta_dot_fft.local_slice()
    my_sphere_parts = [[] for _ in range(p.nbins)]
    for i in range(p.nbins):
        bin_k_min = i * bin_width
        bin_k_max = bin_k_min + bin_width
        for iz in range(my_range[0].start, my_range[0].stop):
            for iy in range(my_range[1].start, my_range[1].stop):
                for ix in range(my_range[2].start, my_range[2].stop):
                    k2 = physical_ks[ix]**2 + physical_ks[iy]**2 + physical_ks[iz]**2
                    if k2 >= bin_k_min**2 and k2 <= bin_k_max**2:
                         my_sphere_parts[i].append((ix, iy, iz))

    sx, sy, sz = screended_theta_dot_fft.subshape
    # P_field(k) = k**2 / L**3 \int d \Omega / 4\pi 0.5 * | field(k) |**2
    spectrum = np.zeros(p.nbins)
    for i in range(p.nbins):
        for (ix, iy, iz) in my_sphere_parts[i]:
            spectrum[i] += np.abs(screended_theta_dot_fft[ix - sx, iy - sy, iz - sz])**2

        spectrum[i] = s.comm.reduce(spectrum[i], op=MPI.SUM, root=s.root)
        if s.rank == s.root:
            spectrum[i] *= surface_element[i]
            spectrum[i] *= 1 / p.L**3 / (4 * np.pi)**2 * 0.5

    if s.rank == s.root:
        return bin_ks, spectrum
    else:
        return None, None
