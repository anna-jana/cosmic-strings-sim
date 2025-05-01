import numpy as np
from mpi4py import MPI
import numba
import AxionStrings

######################################### detecting string cores in the grid #############################################
# string contention method from Moore at al., Axion dark matter: strings and their cores, Appendix A.2
@numba.njit
def crosses_real_axis(psi1 : np.complex128, psi2 : np.complex128) -> bool:
    return np.imag(psi1) * np.imag(psi2) < 0

@numba.njit
def handedness(psi1 : np.complex128, psi2 : np.complex128) -> int:
    return np.sign(np.imag(psi1 * np.conj(psi2)))

@numba.njit
def loop_contains_string(psi1 : np.complex128, psi2 : np.complex128,
        psi3 : np.complex128, psi4 : np.complex128):
    loop = (
          crosses_real_axis(psi1, psi2) * handedness(psi1, psi2)
        + crosses_real_axis(psi2, psi3) * handedness(psi2, psi3)
        + crosses_real_axis(psi3, psi4) * handedness(psi3, psi4)
        + crosses_real_axis(psi4, psi1) * handedness(psi4, psi1)
    )
    return abs(loop) == 2

@numba.njit
def check_if_at_boundary(lnx, lny, lnz, new: np.ndarray[float]):
    return (abs(new[1]) <= np.sqrt(3) or
           abs(new[2]) <= np.sqrt(3) or
           abs(new[3]) <= np.sqrt(3) or
           abs(new[1] - lnx) <= np.sqrt(3) or
           abs(new[2] - lny) <= np.sqrt(3) or
           abs(new[3] - lnz) <= np.sqrt(3))


############################################################### string velocites and gamme factors ###############################################
@numba.njit
def compute_gamma_factors(psi, psi_dot, a, H):
    phi = psi / a
    phi_dot = psi_dot / a**2 - H * phi

    # Moore et al: Axion dark matter: strings and their cores, eq. A10
    c = 0.41238
    gamma2_times_v2 = np.real(np.abs(phi_dot)**2 / c**2 * (1 + np.abs(phi)**2 / (8 * c**2)) +
                              (np.conj(phi) * phi_dot + phi * np.conj(phi_dot))**2 / (16 * c**4))
    # gamma2_times_v2 = v**2 / (1 - v**2) = x
    # v**2 = (1 - v**2) * x
    # x = v**2 + v**2 x = v**2 (1 + x)
    # v**2 = x / (1 + x)
    # v = sqrt(x / (1 + x))
    v2 = gamma2_times_v2 / (1 + gamma2_times_v2)
    v = np.sqrt(v2)
    gamma = np.sqrt(1 / (1 - v**2))

    # weighted by the gamma factor i.e. the energy int gamma ds instead of the string length int ds
    sum_v = v * gamma
    sum_v2 = v2 * gamma
    sum_gamma = gamma**2
    sum_norm = gamma

    return sum_v, sum_v2, sum_gamma, sum_norm


############################################### compute string parameters #######################################################
@numba.njit
def find_string_cores_kernel(a, H, lnx, lny, lnz, psi, psi_dot):
    sum_v = sum_v2 = sum_gamma = sum_norm = 0.0
    string_points = []

    for ix in range(lnx - 1): # - 1 to avoid double counting string cores in different subboxes
        for iy in range(lny - 1):
            for iz in range(lnz - 1):
                i1 = (ix, iy, iz)
                i2 = (ix + 1, iy, iz)
                i3 = (ix + 1, iy + 1, iz)
                i4 = (ix, iy + 1, iz)
                if loop_contains_string(psi[i1], psi[i2], psi[i3], psi[i4]):
                    string_points.append( np.array([ix + 0.5, iy + 0.5, iz]) )
                    new_v_1, new_v2_1, new_gamma_1, new_norm_1 = compute_gamma_factors(psi[i1], psi_dot[i1], a, H)
                    new_v_2, new_v2_2, new_gamma_2, new_norm_2 = compute_gamma_factors(psi[i2], psi_dot[i2], a, H)
                    new_v_3, new_v2_3, new_gamma_3, new_norm_3 = compute_gamma_factors(psi[i3], psi_dot[i3], a, H)
                    new_v_4, new_v2_4, new_gamma_4, new_norm_4 = compute_gamma_factors(psi[i4], psi_dot[i4], a, H)
                    sum_v += new_v_1 + new_v_2 + new_v_3 + new_v_4
                    sum_v2 += new_v2_1 + new_v2_2 + new_v2_3 + new_v2_4
                    sum_gamma += new_gamma_1 + new_gamma_2 + new_gamma_3 + new_gamma_4
                    sum_norm += new_norm_1 + new_norm_2 + new_norm_3 + new_norm_4

                i1 = (ix, iy, iz)
                i2 = (ix, iy + 1, iz)
                i3 = (ix, iy + 1, iz + 1)
                i4 = (ix, iy, iz + 1)
                if loop_contains_string(psi[i1], psi[i2], psi[i3], psi[i4]):
                    string_points.append( np.array([ix, iy + 0.5, iz + 0.5]) )
                    new_v_1, new_v2_1, new_gamma_1, new_norm_1 = compute_gamma_factors(psi[i1], psi_dot[i1], a, H)
                    new_v_2, new_v2_2, new_gamma_2, new_norm_2 = compute_gamma_factors(psi[i2], psi_dot[i2], a, H)
                    new_v_3, new_v2_3, new_gamma_3, new_norm_3 = compute_gamma_factors(psi[i3], psi_dot[i3], a, H)
                    new_v_4, new_v2_4, new_gamma_4, new_norm_4 = compute_gamma_factors(psi[i4], psi_dot[i4], a, H)
                    sum_v += new_v_1 + new_v_2 + new_v_3 + new_v_4
                    sum_v2 += new_v2_1 + new_v2_2 + new_v2_3 + new_v2_4
                    sum_gamma += new_gamma_1 + new_gamma_2 + new_gamma_3 + new_gamma_4
                    sum_norm += new_norm_1 + new_norm_2 + new_norm_3 + new_norm_4

                i1 = (ix, iy, iz)
                i2 = (ix, iy, iz + 1)
                i3 = (ix + 1, iy, iz + 1)
                i4 = (ix + 1, iy, iz)
                if loop_contains_string(psi[i1], psi[i2], psi[i3], psi[i4]):
                    string_points.append( np.array([ix + 0.5, iy, iz + 0.5]) )
                    new_v_1, new_v2_1, new_gamma_1, new_norm_1 = compute_gamma_factors(psi[i1], psi_dot[i1], a, H)
                    new_v_2, new_v2_2, new_gamma_2, new_norm_2 = compute_gamma_factors(psi[i2], psi_dot[i2], a, H)
                    new_v_3, new_v2_3, new_gamma_3, new_norm_3 = compute_gamma_factors(psi[i3], psi_dot[i3], a, H)
                    new_v_4, new_v2_4, new_gamma_4, new_norm_4 = compute_gamma_factors(psi[i4], psi_dot[i4], a, H)
                    sum_v += new_v_1 + new_v_2 + new_v_3 + new_v_4
                    sum_v2 += new_v2_1 + new_v2_2 + new_v2_3 + new_v2_4
                    sum_gamma += new_gamma_1 + new_gamma_2 + new_gamma_3 + new_gamma_4
                    sum_norm += new_norm_1 + new_norm_2 + new_norm_3 + new_norm_4

    return string_points, sum_v, sum_v2, sum_gamma, sum_norm

def find_string_cores(s: AxionStrings.State):
    a = AxionStrings.tau_to_a(s.tau)
    H = AxionStrings.t_to_H(AxionStrings.tau_to_t(s.tau))

    string_points, sum_v, sum_v2, sum_gamma, sum_norm = find_string_cores_kernel(a, H, s.lnx, s.lny, s.lnz, s.psi, s.psi_dot)

    sum_v     = s.comm.reduce(sum_v,     op=MPI.SUM, root=s.root)
    sum_v2    = s.comm.reduce(sum_v2,    op=MPI.SUM, root=s.root)
    sum_gamma = s.comm.reduce(sum_gamma, op=MPI.SUM, root=s.root)
    sum_norm  = s.comm.reduce(sum_norm,  op=MPI.SUM, root=s.root)

    string_points = np.array(string_points)

    if s.rank == s.root:
        mean_v = sum_v / sum_norm
        mean_v2 = sum_v2 / sum_norm
        mean_gamma = sum_gamma / sum_norm
        return string_points, (mean_v, mean_v2, mean_gamma)
    else:
        return string_points, None

##################################################### combine string cores into induvidual strings ###############################################
@numba.njit
def cyclic_dist_squared_1d(x1: float, x2: float, N: int) -> float:
    return min((x1 - x2)**2, (N - x1 + x2)**2, (N - x2 + x1)**2)

@numba.njit
def cyclic_dist_squared(p1 : np.ndarray[float], p2 : np.ndarray[float], N: int) -> float:
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return (
        cyclic_dist_squared_1d(x1, x2, N) +
        cyclic_dist_squared_1d(y1, y2, N) +
        cyclic_dist_squared_1d(z1, z2, N)
    )

def find_induvidual_strings(s: AxionStrings.State, string_points: np.ndarray[float]):
    strings = []
    total_string_length = 0.0

    while string_points.shape[0] > 0:
        # pop string point of the end
        first_point = string_points[-1, :]
        string_points = string_points[:-1, :]

        current_string = [first_point]
        current_length = 1

        while True:
            if len(string_points) == 0:
                # TODO:
                # no string points left in this subboxes and string isnt closed!
                # need to communicate with other subboxes
                # HACK:
                # for now lets collect all string parts seperatly and add halve of the mean between
                # 1 (distance if the string point are on opposite sides of the cube)
                # and (2*(/2)^2)**0.5 = dx / sqrt(2)
                # i.e (1 + 1/sqrt(2)) / 2 (all in units of the lattice spacing)
                total_string_length += (1 + 1 / 2**0.5) / 2
                break

            # find the point in the list of remaining points that is closest
            # to last point in the list of point of our current so far
            dist_to_new = np.inf
            index = None
            for i, point in enumerate(string_points):
                dist = cyclic_dist_squared(current_string[-1], point, s.p.N)
                if dist < dist_to_new:
                    dist_to_new = dist
                    index = i

            # if current_length <= 2 we are still at the beginning of the string
            # first point might be the closest even if the string goes on
            dist_to_first = cyclic_dist_squared(current_string[-1], first_point, s.p.N)
            if current_length <= 2 or dist_to_new < dist_to_first:
                # save point to list of the current string
                current_string.append(string_points[index, :].copy())
                # remove the closest point from the list of remaining points
                string_points[index, :] = string_points[-1, :]
                string_points = string_points[:-1, :]
                # keep track of the total length of the string in this subvolume
                total_string_length += dist_to_new
            else:
                # we close the string if the cloest point in the list of remaining point
                # is farer away that the first point on our string
                # add remaining distance to the first point
                total_string_length += dist_to_first
                strings.append(current_string)
                break

    total_string_length = s.comm.reduce(total_string_length, op=MPI.SUM, root=s.root)

    return total_string_length, strings


