import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import AxionStrings

##################################################### parameter ######################################################
with open("parameter.json", "r") as f:
    p = json.load(f)
    log_start = p["log_start"]
    log_end = p["log_end"]
    L = p["L"]
    N = p["N"]
    Delta_tau = p["Delta_tau"]
    seed = p["seed"]
    k_max = p["k_max"]
    dx = p["dx"]
    tau_start = p["tau_start"]
    tau_end = p["tau_end"]
    tau_span = p["tau_span"]
    nsteps = p["nsteps"]
    nbins = p["nbins"]
    radius = p["radius"]

############################################### different energy components ###############################################
tau, axion_kinetic, axion_gradient, axion_total, radial_kinetic, radial_gradient, radial_potential, radial_total, interaction, total = np.loadtxt("energies.dat", unpack=True)
logs = AxionStrings.tau_to_log(tau)

plt.figure(figsize=(9, 3), constrained_layout=True)
plt.plot(logs, axion_kinetic, color="tab:blue", ls=":", label="axion, kinetic")
plt.plot(logs, axion_gradient, color="tab:blue", ls="--", label="axion, gradient")
plt.plot(logs, axion_total, color="tab:blue", ls="-", label="axion")
plt.plot(logs, radial_kinetic, color="tab:orange", ls=":", label="radial, kinetic")
plt.plot(logs, radial_gradient, color="tab:orange", ls="--", label="radial, gradient")
plt.plot(logs, radial_potential, color="tab:orange", ls="-.", label="radial, potential")
plt.plot(logs, radial_total, color="tab:orange", ls="-", label="radial")
plt.plot(logs, -interaction, color="tab:green", ls="-", label="-interaction = strings")
plt.plot(logs, total, color="black", ls="--", label="total energy density")
plt.yscale("log")
#plt.xlim(plt.xlim()[1], 4.15)
plt.xlabel(r"$log(m_r / H)$")
plt.ylabel(r"averaged energy density $f_a^2 m_r^2$\n")
plt.legend()
plt.savefig("energy_densities.pdf")

plt.figure()
plt.plot(logs, interaction)
plt.xlabel(r"$log(m_r / H)$")
plt.ylabel(r"averaged interaction energy density $f_a^2 m_r^2$\n")
plt.savefig("interaction_term.pdf")

################################################# string length and velocites #############################################
taus, mean_v, mean_v2, mean_gamma = np.loadtxt("velocities.dat", unpack=True)
logs = AxionStrings.tau_to_log(taus)

fig, axs = plt.subplots(3, 1, sharex=True)
fig.subplots_adjust(hspace=0)
axs[0].plot(logs, mean_v)
axs[1].plot(logs, mean_v2)
axs[2].plot(logs, mean_gamma)

axs[2].set_xlabel(r"$\log(m_r / H)$")
axs[0].set_ylabel(r"$\langle v \rangle$")
axs[1].set_ylabel(r"$\langle v^2 \rangle$")
axs[2].set_ylabel(r"$\langle \gamma \rangle$")

plt.savefig("velocties.pdf")

taus, l = np.loadtxt("string_length.dat", unpack=True)
a = AxionStrings.tau_to_a(taus)
t = AxionStrings.tau_to_t(taus)
logs = AxionStrings.tau_to_log(taus)

# string length parameter
# conversion to physical units and factoring out the dilution from the expansion
l_physical = a * dx * l
V_Hubbel = (L * a)**3
zetas = l_physical / V_Hubbel * t**2

# string length pa
plt.figure()
plt.semilogy(logs, zetas)
plt.xlabel(r"$\log(m_r / H)$")
plt.ylabel(r"$\zeta = a l / a^3 L^3 \times t^2$")
plt.savefig("string_length.pdf")

############################################### emission spectra #####################################################
exit()
# compute the instanteous emission spectrum defined in the paper by gorghetto (axion strings: the attractive solution, eq. 33)
def compute_instanteous_emission_spectrum(P1, P2, k1, k2, tau1, tau2):
    # re interpolate such that P1 and P2 have a shared support
    k_min = max(k1[0], k2[0])
    k_max = min(k1[-1], k2[-1])
    ks = np.linspace(k_min, k_max, np.size(k1))

    P1_interp = interp1d(k1, P1)(ks)
    P2_interp = interp1d(k2, P2)(ks)

    # compute time scales
    t1 = AxionStrings.tau_to_t(tau1)
    t2 = AxionStrings.tau_to_t(tau2)
    a1 = AxionStrings.tau_to_a(tau1)
    a2 = AxionStrings.tau_to_a(tau2)
    t_mid = (t2 + t1) / 2
    a_mid = AxionStrings.t_to_a(t_mid)
    log_mid = AxionStrings.tau_to_log(AxionStrings.t_to_tau(t_mid))

    # finite difference
    F = (a2**3 * P2_interp - a1**3 * P1_interp) / (t2 - t1) / a_mid**3

    # normalize
    A = np.sum((F[1:] + F[:-1]) / 2 * (ks[1:] + ks[:-1]) / 2)
    F /= A

    return log_mid, ks, F

def normalize(ks, xs):
    norm = np.sqrt(np.sum(xs**2) * (ks[1] - ks[0]))
    return xs / norm

k1, P1_ppse, P1_uncorrected, P1_screened = np.loadtxt("spectrum1.dat", unpack=True)
k2, P2_ppse, P2_uncorrected, P2_screened = np.loadtxt("spectrum2.dat", unpack=True)

tau1 = tau_start
tau2 = tau_end
log_mid, ks, F_ppse = compute_instanteous_emission_spectrum(P1_ppse, P2_ppse, k1, k2, tau1, tau2)
_, _, F_screened = compute_instanteous_emission_spectrum(P1_screened, P2_screened, k1, k2, tau1, tau2)

log1 = AxionStrings.tau_to_log(tau1)
log2 = AxionStrings.tau_to_log(tau2)

plt.figure()
plt.plot(k1, normalize(k1, P1_ppse), label=f"ppse, log = {log1}")
plt.plot(k2, normalize(k2, P2_ppse), label=f"ppse, log = {log2}")
plt.plot(k1, normalize(k1, P1_screened), label=f"screened, log = {log1}")
plt.plot(k2, normalize(k2, P2_screened), label=f"screeend, log = {log2}")
plt.plot(k1, normalize(k1, P1_uncorrected), label=f"uncorrected, log = {log1}")
plt.plot(k2, normalize(k2, P2_uncorrected), label=f"uncorrected, log = {log2}")
plt.xlabel("physical momentum |k|")
plt.ylabel("power spectrum P(k)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title("spectrum of free axions")
plt.savefig("spectra.pdf")

plt.figure()
plt.loglog(ks, F_ppse, label="ppse, simulation at log=$log_mid")
plt.loglog(ks, F_screened, label="screened, simulation at log=$log_mid")
plt.xlabel("physical momentum |k|")
plt.ylabel("F(k)")
plt.title("instantaneous emission spectrum")
plt.legend()
plt.savefig("instant_emission_spectrum.pdf")
