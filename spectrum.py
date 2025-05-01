import numpy as np
from scipy.linalg import inv
from scipy.fft import rfftfreq, fftfreq
from mpi4py import MPI
from mpi4py_fft import PFFT, DistArray
import numba
import AxionStrings
import energy

################################# helper functions ##################################
@numba.njit
def compute_theta_dot(a : float, psi : np.complex128, psi_dot : np.complex128):
    R = np.real(psi)
    I = np.imag(psi)
    R_dot = np.real(psi_dot)
    I_dot = np.imag(psi_dot)
    d_theta_d_tau = (I_dot * R - I * R_dot) / (R**2 - I**2)
    return d_theta_d_tau / a

@numba.njit
def calc_k_max_grid(N, d):
    if N % 2 == 0:
        return 2 * np.pi * (N / 2) / (d*N)
    else:
        return 2 * np.pi * ((N - 1) / 2) / (d*N)

# k = 2pi * integer_k / (Delta x * N)
@numba.njit
def single_index_to_integer_k(N:int, i:int):
    if N % 2 == 0:
        return i if i < N // 2 else i - N
    else:
        return i if i <= (N - 1) // 2 else i - N

@numba.njit
def single_integer_k_to_index(N:int, integer_k:int) -> int:
    return integer_k if integer_k >= 0 else integer_k + N

@numba.njit
def index_to_integer_k(N:int, i:(int, int, int)) -> (int, int, int):
    return (
        single_index_to_integer_k(N, i[0]),
        single_index_to_integer_k(N, i[1]),
        single_index_to_integer_k(N, i[2]),
    )

@numba.njit
def integer_k_to_index(N:int, integer_k:(int, int, int)) -> (int, int, int):
    ix = single_integer_k_to_index(N, integer_k[1])
    if ix >= N // 2 + 1:
        ix -= N // 2 + 1
    return (
        ix,
        single_integer_k_to_index(N, integer_k[2]),
        single_integer_k_to_index(N, integer_k[3]),
    )

@numba.njit
def min_integer_k(N):
    return -N // 2 if N % 2 == 0 else -(N - 1) // 2

@numba.njit
def max_integer_k(N):
    return (N - 1) // 2

@numba.njit
def single_check_integer_k(integer_k:int, min_l, max_l):
    return integer_k > max_l or integer_k < min_l

@numba.njit
def check_integer_k(integer_k:(int, int, int), min_l, max_l) -> bool:
    return (single_check_integer_k(integer_k[0], min_l, max_l) and
           single_check_integer_k(integer_k[1], min_l, max_l) and
           single_check_integer_k(integer_k[2], min_l, max_l))

@numba.njit
def substract_wave_numbers_lookup(W_fft, p, min_l, max_l, i1, i2):
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
        return  W_fft[ix, iy, iz]

########################################### distributed ffts ###############################################
def distributed_fft(s: AxionStrings.State, array):
    darray = DistArray((s.p.N, s.p.N, s.p.N), [0, 1, 2], buffer=array, dtype=np.float64,)
    plan = PFFT(s.comm, darray=darray, backend="numpy")
    ans = plan.forward(darray)
    darray_ans = DistArray((s.p.N, s.p.N, s.p.N // 2),
            [0, 1, 2], buffer=ans, dtype=np.complex128,)

    return darray_ans

################################### compute a powerspectrum of an arbitary array ####################################
@numba.njit
def compute_spheres_kernel(i, N, bin_width, physical_ks_short, physical_ks_long):
    spheres = []
    bin_k_min = i * bin_width
    bin_k_max = bin_k_min + bin_width
    for ix in range(N):
        for iy in range(N):
            for iz in range(N // 2 + 1):
                k2 = physical_ks_long[ix]**2 + physical_ks_long[iy]**2 + physical_ks_short[iz]**2
                if k2 >= bin_k_min**2 and k2 <= bin_k_max**2:
                    spheres.append((ix, iy, iz))
    return spheres

def power_spectrum_utils(p: AxionStrings.Parameter, a):
    # prepare histogram
    dx_physical = p.dx * a
    kmax = calc_k_max_grid(p.N, dx_physical)
    physical_ks_short = rfftfreq(p.N, dx_physical) * 2*np.pi
    physical_ks_long = fftfreq(p.N, dx_physical) * 2*np.pi

    Delta_k = 2*np.pi / (p.N * dx_physical)
    bin_width = kmax / p.nbins

    i = np.arange(p.nbins)
    bin_ks = i * bin_width + bin_width/2.0

    vol = 4.0/3.0 * np.pi * (((i + 1)*bin_width)**3 - (i*bin_width)**3)
    area = 4*np.pi * (i*bin_width + bin_width/2)**2
    surface_element = area / vol * Delta_k**3 / bin_ks**2

    spheres = [compute_spheres_kernel(i, p.N, bin_width, physical_ks_short, physical_ks_long) for i in range(p.nbins)]

    return physical_ks_short, physical_ks_long, bin_width, surface_element, bin_ks, spheres

# P_field(k) = k**2 / L**3 \int d \Omega / 4\pi 0.5 * | field(k) |**2
@numba.njit
def compute_power_spectrum_kernel(field_fft, i, bin_width,
        start_x, stop_x, start_y, stop_y, start_z, stop_z,
        physical_ks_short, physical_ks_long, sx, sy, sz):
    s = 0.0
    bin_k_min = i * bin_width
    bin_k_max = bin_k_min + bin_width
    for ix in range(start_x, stop_x):
        for iy in range(start_y, stop_y):
            for iz in range(start_z, stop_z):
                k2 = physical_ks_long[ix]**2 + physical_ks_long[iy]**2 + physical_ks_short[iz]**2
                if k2 >= bin_k_min**2 and k2 <= bin_k_max**2:
                    s += np.abs(field_fft[ix - sx, iy - sy, iz - sz])**2
    return s

def compute_power_spectrum(s: AxionStrings.State, field, spheres, surface_element, physical_ks_short, physical_ks_long, bin_width):
    field_fft = distributed_fft(s, field)
    sx, sy, sz = field_fft.substart
    my_range = field_fft.local_slice()
    start_x, stop_x = my_range[0].start, my_range[0].stop
    start_y, stop_y = my_range[1].start, my_range[1].stop
    start_z, stop_z = my_range[2].start, my_range[2].stop

    spectrum = np.zeros(s.p.nbins)
    for i in range(s.p.nbins):
        # part of the powerspectrum integration for this subbox
        local_s = compute_power_spectrum_kernel(field_fft, i, bin_width, start_x, stop_x,
                start_y, stop_y, start_z, stop_z, physical_ks_short, physical_ks_long, sx, sy, sz)
        global_s = s.comm.reduce(local_s, op=MPI.SUM, root=s.root)
        if s.rank == s.root:
            spectrum[i] = global_s * surface_element[i] * 1 / s.p.L**3 / (4 * np.pi)**2 * 0.5

    if s.rank == s.root:
        return spectrum
    else:
        return None

##################################### pseudo power spectrum estimator (ppse) #######################################
# TODO: adapt for mpi
# TODO: numba?
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

# TODO: numba?
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

    physical_ks_short, physical_ks_long, bin_width, surface_element, bin_ks, spheres = power_spectrum_utils(p, a)

    spectrum_uncorrected = compute_power_spectrum(s, p, theta_dot, spheres, surface_element, physical_ks_short, physical_ks_long, bin_width)

    W_fft = distributed_fft(s, W)

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

##################################################### autoscreen powerspectrum #############################################
def compute_spectrum_autoscreen(s: AxionStrings.State):
    a = AxionStrings.tau_to_a(s.tau)
    theta_dot = compute_theta_dot(a, s.psi[1:-1, 1:-1, 1:-1], s.psi_dot[1:-1, 1:-1, 1:-1])
    r = energy.compute_radial_mode(s.psi[1:-1, 1:-1, 1:-1], a)
    screened_theta_dot = (1 + r) * theta_dot

    physical_ks_short, physical_ks_long, bin_width, surface_element, bin_ks, spheres = power_spectrum_utils(s.p, a)
    spectrum = compute_power_spectrum(s, screened_theta_dot, spheres, surface_element, physical_ks_short, physical_ks_long, bin_width)

    return (
        (bin_ks, spectrum) if s.rank == s.root else (None, None)
    )
