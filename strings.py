import operator
import numpy as np
from mpi4py import MPI
import AxionStrings

# string contention method from Moore at al., Axion dark matter: strings and their cores, Appendix A.2
def crosses_real_axis(psi1 : np.complex128, psi2 : np.complex128) -> bool:
    return np.imag(psi1) * np.imag(psi2) < 0

def handedness(psi1 : np.complex128, psi2 : np.complex128) -> int:
    return np.sign(np.imag(psi1 * np.conj(psi2)))

def loop_contains_string(psi1 : np.complex128, psi2 : np.complex128,
        psi3 : np.complex128, psi4 : np.complex128):
    loop = (
          crosses_real_axis(psi1, psi2) * handedness(psi1, psi2)
        + crosses_real_axis(psi2, psi3) * handedness(psi2, psi3)
        + crosses_real_axis(psi3, psi4) * handedness(psi3, psi4)
        + crosses_real_axis(psi4, psi1) * handedness(psi4, psi1)
    )
    return abs(loop) == 2

def physical_total_string_length(s:AxionStrings.State, p:AxionStrings.Parameter, l:float):
    a = AxionStrings.tau_to_a(s.tau)
    t = AxionStrings.tau_to_t(s.tau)
    return a * l / (p.L**3 * a**3) * t**2

def total_string_length(s:AxionStrings.State, p:AxionStrings.Parameter, strings:list[list[np.ndarray[float]]]):
    l = p.dx * sum(
        sum(np.sqrt(cyclic_dist_squared(p, s[i], s[(i + 1) % len(s)])) for i in range(len(s)))
        )
    return physical_total_string_length(s, p, l)

def cyclic_dist_squared_1d(p : AxionStrings.Parameter, x1 : float, x2 : float) -> float:
    return min((x1 - x2)**2, (p.N - x1 + x2)**2, (p.N - x2 + x1)**2)

def cyclic_dist_squared(p : AxionStrings.Parameter, p1 : np.ndarray[float], p2 : np.ndarray[float]) -> float:
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return (
        cyclic_dist_squared_1d(p, x1, x2) +
        cyclic_dist_squared_1d(p, y1, y2) +
        cyclic_dist_squared_1d(p, z1, z2)
    )

c = 0.41238

def compute_gamma_factors(s:AxionStrings.State, i1, i2, i3, i4):
    sum_v2 = 0.0
    sum_v = 0.0
    sum_gamma = 0.0
    sum_norm = 0.0

    for i in (i1, i2, i3, i4):
        a = AxionStrings.tau_to_a(s.tau)
        H = AxionStrings.t_to_H(AxionStrings.tau_to_t(s.tau))

        psi = s.psi[i]
        psi_dot = s.psi_dot[i]

        phi = psi / a
        phi_dot = psi_dot / a**2 - H * phi

        # Moore et al: Axion dark matter: strings and their cores, eq. A10
        gamma2_times_v2 = np.real(np.abs2(phi_dot) / c**2 * (1 + np.abs2(phi) / (8 * c**2)) + (phi.conj() * phi_dot + phi * phi_dot.conj())**2 / (16 * c**4))
        # gamma2_times_v2 = v**2 / (1 - v**2) = x
        # v**2 = (1 - v**2) * x
        # x = v**2 + v**2 x = v**2 (1 + x)
        # v**2 = x / (1 + x)
        # v = sqrt(x / (1 + x))
        v2 = gamma2_times_v2 / (1 + gamma2_times_v2)
        v = np.sqrt(v2)
        gamma = np.sqrt(1 / (1 - v**2))

        # weighted by the gamma factor i.e. the energy int gamma ds instead of the string length int ds
        sum_v += v * gamma
        sum_v2 += v2 * gamma
        sum_gamma += gamma**2
        sum_norm += gamma

    return sum_v, sum_v2, sum_gamma, sum_norm

def dist(p1, p2): return np.linalg.norm(p1 - p2)

def check_if_at_boundary(s:AxionStrings.State, new:np.ndarray[float]):
    return (abs(new[1]) <= np.sqrt(3) or
           abs(new[2]) <= np.sqrt(3) or
           abs(new[3]) <= np.sqrt(3) or
           abs(new[1] - s.lnx) <= np.sqrt(3) or
           abs(new[2] - s.lny) <= np.sqrt(3) or
           abs(new[3] - s.lnz) <= np.sqrt(3))

def detect_strings(s:AxionStrings.State, p:AxionStrings.Parameter):
    string_points = set()
    points = []
    string_length = 0.0
    sum_v = 0.0
    sum_v2 = 0.0
    sum_gamma = 0.0
    sum_norm = 0.0

    for ix in range(s.lnx):
        for iy in range(s.lny):
            for iz in range(s.lnz):
                i1 = (ix, iy, iz)
                i2 = (ix + 1, iy, iz)
                i3 = (ix + 1, iy + 1, iz)
                i4 = (ix, iy + 1, iz)
                if loop_contains_string(s.psi[i1], s.psi[i2], s.psi[i3], s.psi[i4]):
                    new = np.array([ix - 1 + 0.5, iy - 1 + 0.5, iz])
                    if check_if_at_boundary(s, new):
                        string_length += 3/2
                    string_points.add(new)
                    new_v, new_v2, new_gamma, new_norm = compute_gamma_factors(s, i1, i2, i3, i4)
                    sum_v += new_v
                    sum_v2 += new_v2
                    sum_gamma += new_gamma
                    sum_norm += new_norm

                i1 = (ix, iy, iz)
                i2 = (ix, iy + 1, iz)
                i3 = (ix, iy + 1, iz + 1)
                i4 = (ix, iy, iz + 1)
                if loop_contains_string(s.psi[i1], s.psi[i2], s.psi[i3], s.psi[i4]):
                    new = np.array([ix, iy - 1 + 0.5, iz - 1 + 0.5])
                    if check_if_at_boundary(s, new):
                        string_length += 3/2
                    string_points.add(new)
                    new_v, new_v2, new_gamma, new_norm = compute_gamma_factors(s, i1, i2, i3, i4)
                    sum_v += new_v
                    sum_v2 += new_v2
                    sum_gamma += new_gamma
                    sum_norm += new_norm

                i1 = (ix, iy, iz)
                i2 = (ix, iy, iz + 1)
                i3 = (ix + 1, iy, iz + 1)
                i4 = (ix + 1, iy, iz)
                if loop_contains_string(s.psi[i1], s.psi[i2], s.psi[i3], s.psi[i4]):
                    new = np.array([ix - 1 + 0.5, iy, iz - 1 + 0.5])
                    if check_if_at_boundary(s, new):
                        string_length += 3/2
                    string_points.add(new)
                    new_v, new_v2, new_gamma, new_norm = compute_gamma_factors(s, i1, i2, i3, i4)
                    sum_v += new_v
                    sum_v2 += new_v2
                    sum_gamma += new_gamma
                    sum_norm += new_norm

    while len(string_points) != 0:
        first_point = string_points.pop()
        points.append(first_point)
        last_point = first_point
        current_length = 1

        while True:
            if len(string_points) == 0:
                # in mpi we leaf out the warning
                # if cyclic_dist_squared(p, last_point, first_point) >= sqrt(3)
                #     @warn "no points left but string isnt closed"
                # end
                break

            closest = np.argmin([dist(last_point, point) for point in string_points]) # TODO slow
            dist_to_new = dist(last_point, closest)
            dist_to_first = dist(last_point, first_point)

            if current_length <= 2 or dist_to_new < dist_to_first:
                string_points.remove(closest)
                points.append(closest)
                last_point = closest
                string_length += dist_to_new
            else:
                break # we closed the string
                string_length += dist_to_first

    sum_v = MPI.Reduce(sum_v, operator.add, s.root, s.comm)
    sum_v2 = MPI.Reduce(sum_v2, operator.add, s.root, s.comm)
    sum_gamma = MPI.Reduce(sum_gamma, operator.add, s.root, s.comm)
    sum_norm = MPI.Reduce(sum_norm, operator.add, s.root, s.comm)

    if s.rank == s.root:
        mean_v = sum_v / sum_norm
        mean_v2 = sum_v2 / sum_norm
        mean_gamma = sum_gamma / sum_norm
    else:
        mean_v = mean_v2 = mean_gamma = 0.0

    string_length = MPI.Reduce(string_length, operator.add, s.root, s.comm)

    return total_string_length(s, p, string_length), points, mean_v, mean_v2, mean_gamma
