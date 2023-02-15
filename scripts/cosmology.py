import numpy as np

def log_to_H(log):
    return 1/np.exp(log)

def H_to_t(H):
    return 1 / (2*H)

def t_to_H(t):
    return 1 / (2*t)

def H_to_log(H):
    return np.log(1/H)

def t_to_tau(t):
    return -2*np.sqrt(t)

def log_to_tau(log):
    return t_to_tau(H_to_t(log_to_H(log)))

def t_to_a(t):
    return np.sqrt(t)

def tau_to_t(tau):
    return (-0.5*(tau))**2

def tau_to_a(tau):
    return -0.5*tau

def tau_to_log(tau):
    return H_to_log(t_to_H(tau_to_t(tau)))

# lets say we want to simulat until this log
def minimal_parameters(final_log):
    # where
    # log = log(m_r / H)

    # minial length required for the simulation to contain one hubble patch at the end of the simulation
    # during the simulation it contains more than one (a patch is smaller)
    L_min = 1 / log_to_H(final_log)

    # we need at least one grid point at the end of the simulation in a string core i.e. 1/m_r or 1 in 1/m_r (code) units
    # our spacial coordinates are comoving, hence the physical lattice spacing:
    # dx_physical = dx_comoving * a(t)
    # dx_comoving = L / N
    # 1 / dx_physical > 1
    # dx_physical < 1
    # dx_comoving * a(t) < 1
    # L / N * a(t) < 1
    # L * a(t) < N
    # has to hold for all t and is the tightest for larger a, i.e. log_end
    # N_min = L * a(log_end)
    N_min = int(np.ceil(L_min * tau_to_a(log_to_tau(final_log))))

    print(f"for final log = {final_log = } we need at least {L_min = } and {N_min = }")
    return L_min, N_min
