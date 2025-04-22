import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def log_to_H(l): return 1.0 / np.exp(l)
def H_to_t(H): return 1 / (2*H)
def t_to_H(t): return 1 / (2*t)
def H_to_log(H): return np.log(1/H)
def t_to_tau(t): return -2*np.sqrt(t)
def log_to_tau(log): return t_to_tau(H_to_t(log_to_H(log)))
def t_to_a(t): return np.sqrt(t)
def tau_to_t(tau): return -0.5*(tau)**2
def tau_to_a(tau): return -0.5*tau
def tau_to_log(tau): return H_to_log(t_to_H(tau_to_t(tau)))

# string movie
# function plot_strings(params :: AxionStrings.Parameter, strings; colors_different=False)
#     fig = gcf()
#     fig.add_subplot(projection="3d")
#
#     for string in strings
#         xs = [string[1][1]]
#         ys = [string[1][2]]
#         zs = [string[1][3]]
#         prev = string[1]
#         color = nothing
#
#         for p in string[2:end]
#             if norm(p .- prev) <= sqrt(3)
#                 push!(xs, p[1])
#                 push!(ys, p[2])
#                 push!(zs, p[3])
#             else
#                 l, = plot(xs .* params.dx, ys .* params.dx, zs .* params.dx, color=colors_different ? color : "tab:blue")
#                 color = l.get_color()
#                 xs = [p[1]]
#                 ys = [p[2]]
#                 zs = [p[3]]
#             end
#             prev = p
#         end
#
#         if norm(string[1] - string[end]) <= sqrt(3)
#             push!(xs, string[1][1])
#             push!(ys, string[1][2])
#             push!(zs, string[1][3])
#         end
#
#         plot(xs .* params.dx, ys .* params.dx, zs .* params.dx, color=colors_different ? color : "tab:blue")
#     end
#
#     xlabel(r"$x m_r$")
#     ylabel(r"$y m_r$")
#     zlabel(r"$z m_r$")
#
#     return nothing
# end
#
# function make_string_movie(p :: AxionStrings.Parameter)
#     tmpdir = tempname()
#     mkdir(tmpdir)
#     files = String[]
#     strings = JSON.parse(read("strings.json", String))
#     ioff()
#
#     println("plotting all frames")
#     figure()
#     for (i, (tau, any_strs)) in enumerate(strings)
#         local strs = convert(Vector{Vector{SVector{3, Float64}}}, any_strs)
#         println("$i of $(length(strings))")
#         clf()
#         plot_strings(p, strs; colors_different=False)
#         title(r"$\tau =$" * (@sprintf "%.2f" tau) * r", $\log(m_r/H) = $" *
#               (@sprintf "%.2f" AxionStrings.tau_to_log(tau)))
#         fname = joinpath(tmpdir, "strings_step=$i.jpg")
#         savefig(fname)
#         push!(files, fname)
#     end
#
#     plt.close("all")
#     ion()
#     println("creating gif")
#     outfile = "strings.gif"
#     run(Cmd(vcat(["convert", "-delay", "20", "-loop", "0"], files, outfile)))
#     println("done")
# end

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

data = np.readtxt("energies.dat")
(tau, axion_kinetic, axion_gradient, axion_total,
   radial_kinetic, radial_gradient, radial_potential,
   radial_total, interaction, total) = data.T
logs = tau_to_log(tau)

plt.figure(figsize=(9, 3), constrained_layout=True)
plt.plot(logs, axion_kinetic, color="tab:blue", ls=":", label="axion, kinetic")
plt.plot(logs, axion_gradient, color="tab:blue", ls="--", label="axion, gradient")
plt.plot(logs, axion_total, color="tab:blue", ls="-", label="axion")
plt.plot(logs, radial_kinetic, color="tab:orange", ls=":", label="radial, kinetic")
plt.plot(logs, radial_gradient, color="tab:orange", ls="--", label="radial, gradient")
plt.plot(logs, radial_potential, color="tab:orange", ls="-.", label="radial, potential")
plt.plot(logs, radial_total, color="tab:orange", ls="-", label="radial")
plt.plot(logs, interaction, color="tab:green", ls="-", label="interaction = strings")
plt.plot(logs, total, color="black", ls="--", label="total energy density")
# plot(logs, axion_kinetic ./ total, color="tab:blue", ls="-", label="axion, kinetic")
# plot(logs, axion_gradient ./ total, color="tab:blue", ls="--", label="axion, gradient")
# plot(logs, axion_total ./ total, color="tab:blue", ls="-", lw=2, label="axion")
# plot(logs, radial_kinetic ./ total, color="tab:orange", ls="-", label="radial, kinetic")
# plot(logs, radial_gradient ./ total, color="tab:orange", ls="--", label="radial, gradient")
# plot(logs, radial_potential ./ total, color="tab:orange", ls=":", label="radial, potential")
# plot(logs, radial_total ./ total, color="tab:orange", ls="-", lw=2, label="radial")
# plot(logs, interaction ./ total, color="tab:green", ls="-", label="interaction axion and radial = strings")
plt.yscale("log")
plt.xlim(plt.xlim()[1], 4.15)
plt.xlabel(r"$log(m_r / H)$")
plt.ylabel(r"averaged energy density $f_a^2 m_r^2$\n")
plt.legend()
plt.savefig("energy_densities.pdf")

plt.figure()
plt.plot(logs, interaction)
plt.xlabel(r"$log(m_r / H)$")
plt.ylabel(r"averaged interaction energy density $f_a^2 m_r^2$\n")
plt.savefig("interaction_term.pdf")

data = np.readtxt("velocities.dat")
mean_v, mean_v2, mean_gamma = data[:, 1], data[:, 2], data[:, 3]

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

data = np.readtxt("string_length.dat")
taus, zeta = data.T
logs = tau_to_log(taus)

plt.figure()
plt.plot(logs, zeta)
plt.xlabel(r"$\log(m_r / H)$")
plt.ylabel(r"$\zeta = a l / a^3 L^3 \times t^2$")
plt.savefig("string_length.pdf")

data = np.readtxt("spectrum1.dat")
k1, P1_ppse, P1_uncorrected, P1_screened = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
data = np.readtxt("spectrum2.dat")
k2, P2_ppse, P2_uncorrected, P2_screened = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
tau1 = p.Delta_tau * (p.nsteps - 1)
tau2 = p.Delta_tau * p.nsteps

log1 = tau_to_log(tau1)
log2 = tau_to_log(tau2)

# compute the instanteous emission spectrum defined in the paper by gorghetto (axion strings: the attractive solution, eq. 33)
def compute_instanteous_emission_spectrum(P1, P2, k1, k2, tau1, tau2):
    # re interpolate such that P1 and P2 have a shared support
    k_min = max(k1[0], k2[0])
    k_max = min(k1[-1], k2[-1])
    ks = np.linspace(k_min, k_max, np.size(k1))

    P1_interp = interp1d(k1, P1)(ks)
    P2_interp = interp1d(k2, P2)(ks)

    # compute time scales
    t1 = tau_to_t(tau1)
    t2 = tau_to_t(tau2)
    a1 = tau_to_a(tau1)
    a2 = tau_to_a(tau2)
    t_mid = (t2 + t1) / 2
    a_mid = t_to_a(t_mid)
    log_mid = tau_to_log(t_to_tau(t_mid))

    # finite difference
    F = (a2**3 * P2_interp - a1**3 * P1_interp) / (t2 - t1) / a_mid**3

    # normalize
    A = np.sum((F[1:] + F[:-1]) / 2 * (ks[1:] + ks[:-1]) / 2)
    F /= A

    return log_mid, ks, F

log_mid, ks, F_ppse = compute_instanteous_emission_spectrum(P1_ppse, P2_ppse, k1, k2, tau1, tau2)
_, _, F_screened = compute_instanteous_emission_spectrum(P1_screened, P2_screened, k1, k2, tau1, tau2)

#F_fit_ppse = fit(log.(ks[1:end-1]), log.(F_ppse[1:end-1]), 1)
#F_fit_screened = fit(log.(ks[1:end-1]), log.(F_screened[1:end-1]), 1)
#q_fit_ppse = -F_fit_ppse[1]
#q_fit_screened = -F_fit_screened[1]

def normalize(ks, xs):
    norm = np.sqrt(np.sum(xs**2) * (ks[1] - ks[0]))
    return xs / norm

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
#plt.loglog(ks[1:end-1], np.exp(F_fit_ppse(log(ks[:-1]))), label=f"ppse, fit q = {q_fit_ppse}")
#plt.loglog(ks[1:end-1], np.exp(F_fit_screened(log(ks[0:-1]))), label=f"screened, fit q = {e_fit_screened}")
plt.xlabel("physical momentum |k|")
plt.ylabel("F(k)")
plt.title("instantaneous emission spectrum")
plt.legend()
plt.savefig("instant_emission_spectrum.pdf")
