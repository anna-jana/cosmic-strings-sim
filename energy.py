import numpy as np
from mpi4py import MPI
import numba
import AxionStrings

@numba.njit
def compute_energy_at(a, H, dx, psi, psi_dot,
        theta_left, theta_right, theta_front, theta_back, theta_top, theta_bottom,
        radial, radial_left, radial_right, radial_front, radial_back, radial_top, radial_bottom,
    ):
    R = np.real(psi)
    I = np.imag(psi)
    R_dot = np.real(psi_dot)
    I_dot = np.imag(psi_dot)

    ## axion
    # kinetic
    d_theta_d_tau = (I_dot * R - I * R_dot) / abs(psi)
    axion_kinetic = 0.5 / a**2 * d_theta_d_tau**2
    # gradient
    diff_theta_x = theta_left - theta_right
    diff_theta_y = theta_front - theta_back
    diff_theta_z = theta_top - theta_bottom
    axion_gradient = 0.5 / (dx * 2)**2 * (diff_theta_x**2 + diff_theta_y**2 + diff_theta_z**2) / a**2

    ## radial mode
    # kinetic
    d_abs_psi_d_tau = (R * R_dot + I * I_dot) / abs(psi)
    abs_psi = abs(psi)
    d_r_d_tau = np.sqrt(2) * (d_abs_psi_d_tau / a - H * abs_psi)
    radial_kinetic = 0.5 / a**2 * d_r_d_tau**2 # 0.5 (dr/dt)**2

    # gradient
    diff_radial_x = radial_left - radial_right
    diff_radial_y = radial_front - radial_back
    diff_radial_z = radial_top - radial_bottom
    radial_gradient = 0.5 / (dx * 2)**2 * (diff_radial_x**2 + diff_radial_y**2 + diff_radial_z**2) / a**2

    # potential
    inner = radial**2 + 2.0 * radial
    radial_potential = inner**2 / 8.0

    ## interaction
    interaction = inner / 2 * (axion_kinetic + axion_gradient)

    return axion_kinetic, axion_gradient, radial_kinetic, radial_gradient, radial_potential, interaction

@numba.njit
def compute_radial_mode(psi, a):
    # return sqrt(2) * abs(psi) / a - 1
    return np.sqrt(2) * np.real(psi / np.exp(np.angle(psi) * 1j)) / a - 1

@numba.njit
def compute_energy_local(lnx, lny, lnz, theta, radial, psi, psi_dot, a, H, dx):
    mean_axion_kinetic = 0.0
    mean_axion_gradient = 0.0
    mean_radial_kinetic = 0.0
    mean_radial_gradient = 0.0
    mean_radial_potential = 0.0
    mean_interaction = 0.0

    for ix in range(1, lnx + 1):
       for iy in range(1, lny + 1):
            for iz in range(1, lnz + 1):
                theta_left, theta_right = theta[ix + 1, iy, iz], theta[ix - 1, iy, iz]
                theta_front, theta_back = theta[ix, iy + 1, iz], theta[ix, iy - 1, iz]
                theta_top, theta_bottom = theta[ix, iy, iz + 1], theta[ix, iy, iz - 1]

                radial_left, radial_right = radial[ix + 1, iy, iz], radial[ix - 1, iy, iz]
                radial_front, radial_back = radial[ix, iy + 1, iz], radial[ix, iy - 1, iz]
                radial_top, radial_bottom = radial[ix, iy, iz + 1], radial[ix, iy, iz - 1]
                r = radial[ix, iy, iz]

                axion_kinetic, axion_gradient, radial_kinetic, radial_gradient, radial_potential, interaction = compute_energy_at(
                    a, H, dx, psi[ix, iy, iz], psi_dot[ix, iy, iz],
                    theta_left, theta_right, theta_front, theta_back, theta_top, theta_bottom,
                    r, radial_left, radial_right, radial_front, radial_back, radial_top, radial_bottom)

                mean_axion_kinetic += axion_kinetic
                mean_axion_gradient += axion_gradient
                mean_radial_kinetic += radial_kinetic
                mean_radial_gradient += radial_gradient
                mean_radial_potential += radial_potential
                mean_interaction += interaction

    return mean_axion_kinetic, mean_axion_gradient, mean_radial_kinetic, mean_radial_gradient, mean_radial_potential, mean_interaction

def compute_energy(s: AxionStrings.State, p: AxionStrings.Parameter):
    a = AxionStrings.tau_to_a(s.tau)
    H = AxionStrings.t_to_H(AxionStrings.tau_to_t(s.tau))

    s.exchange_field()

    theta = np.angle(s.psi)
    radial = compute_radial_mode(s.psi, a)

    mean_axion_kinetic, mean_axion_gradient, mean_radial_kinetic, mean_radial_gradient, mean_radial_potential, mean_interaction = \
        compute_energy_local(s.lnx, s.lny, s.lnz, theta, radial, s.psi, s.psi_dot, a, H, p.dx)

    # sum all the subboxes
    mean_axion_kinetic    = s.comm.reduce(mean_axion_kinetic,    op=MPI.SUM, root=s.root)
    mean_axion_gradient   = s.comm.reduce(mean_axion_gradient,   op=MPI.SUM, root=s.root)
    mean_radial_kinetic   = s.comm.reduce(mean_radial_kinetic,   op=MPI.SUM, root=s.root)
    mean_radial_gradient  = s.comm.reduce(mean_radial_gradient,  op=MPI.SUM, root=s.root)
    mean_radial_potential = s.comm.reduce(mean_radial_potential, op=MPI.SUM, root=s.root)
    mean_interaction      = s.comm.reduce(mean_interaction,      op=MPI.SUM, root=s.root)

    if s.rank == s.root:
        mean_axion_kinetic /= p.N**3
        mean_axion_gradient /= p.N**3
        mean_radial_kinetic /= p.N**3
        mean_radial_gradient /= p.N**3
        mean_radial_potential /= p.N**3
        mean_interaction /= p.N**3

        mean_axion_total = mean_axion_kinetic + mean_axion_gradient
        mean_radial_total = mean_radial_kinetic + mean_radial_gradient + mean_radial_potential
        mean_total = mean_axion_total + mean_radial_total + mean_interaction

        return (mean_axion_kinetic, mean_axion_gradient, mean_axion_total,
            mean_radial_kinetic, mean_radial_gradient, mean_radial_potential, mean_radial_total,
            mean_interaction, mean_total)

    else:
        return None

