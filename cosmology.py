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

log_start = 2
log_end = 2.5
tau_start = log_to_tau(log_start)
tau_end = log_to_tau(log_end)
dtau = -1e-2
