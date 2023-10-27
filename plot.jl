println("loading packages")

include("AxionStrings.jl")

using PyPlot
using DelimitedFiles
using Interpolations
using JSON
using Polynomials
using Printf
using StaticArrays

println("start")

ioff()

p = JSON.parse(read("parameter.json", String)) # load json
p = Dict([Symbol(s) => v for (s, v) in p]) # string keys -> symbols
p = AxionStrings.Parameter(; p...) # create type

# energy densities
println("energy")
data = readdlm("energies.dat")
(tau, axion_kinetic, axion_gradient, axion_total,
   radial_kinetic, radial_gradient, radial_potential,
   radial_total, interaction, total) = tuple([data[:, i] for i in 1:size(data, 2)]...)
logs = AxionStrings.tau_to_log.(tau)
figure()
plot(logs, axion_kinetic, color="tab:blue", ls="-", label="axion, kinetic")
plot(logs, axion_gradient, color="tab:blue", ls="--", label="axion, gradient")
plot(logs, axion_total, color="tab:blue", ls="-", lw=2, label="axion")
plot(logs, radial_kinetic, color="tab:orange", ls="-", label="radial, kinetic")
plot(logs, radial_gradient, color="tab:orange", ls="--", label="radial, gradient")
plot(logs, radial_potential, color="tab:orange", ls=":", label="radial, potential")
plot(logs, radial_total, color="tab:orange", ls="-", lw=2, label="radial")
plot(logs, interaction, color="tab:green", ls="-", label="interaction axion and radial = strings")
plot(logs, total, color="black", ls="-", lw=2, label="total energy density")
yscale("log")
xlabel(raw"$log(m_r / H)$")
ax = gca()
ax.invert_xaxis()
ylabel(raw"averaged energy density $f_a^2 m_r^2$\n")
legend()
savefig("energy_densities.pdf")

# spectra
println("spectra")
data = readdlm("spectrum1.dat")
k1, P1 = data[:, 1], data[:, 2]
data = readdlm("spectrum2.dat")
k2, P2 = data[:, 1], data[:, 2]
tau1 = p.Delta_tau * (p.nsteps - 1)
tau2 = p.Delta_tau * p.nsteps

log1 = AxionStrings.tau_to_log(tau1)
log2 = AxionStrings.tau_to_log(tau2)
t1 = AxionStrings.tau_to_t(tau1)
t2 = AxionStrings.tau_to_t(tau2)
a1 = AxionStrings.tau_to_a(tau1)
a2 = AxionStrings.tau_to_a(tau2)

k_min = max(k1[1], k2[1])
k_max = min(k1[end], k2[end])
ks = range(k_min, k_max, length=length(k1))
P1_interp = linear_interpolation(k1, P1)(ks)
P2_interp = linear_interpolation(k2, P2)(ks)

t_mid = (t2 + t1) / 2
a_mid = AxionStrings.t_to_a(t_mid)
log_mid = AxionStrings.tau_to_log(AxionStrings.t_to_tau(t_mid))

F = @. (a2^3 * P2_interp - a1^3 * P1_interp) / (t2 - t1) / a_mid^3

A = sum(@. (F[2:end] + F[1:end-1]) / 2 * (ks[2:end] + ks[1:end-1]) / 2)
F ./= A

F_fit = fit(log.(ks[1:end-1]), log.(F[1:end-1]), 1)
q_fit = -F_fit[1]

figure()
plot(k1, P1, label="log = $log1")
plot(k2, P2, label="log = $log2")
xlabel("comoving momentum |k|")
ylabel("power spectrum P(k)")
xscale("log")
yscale("log")
legend()
title("spectrum of free axions")
savefig("spectra.pdf")

figure()
loglog(ks, F, label="simulation at log=$log_mid")
loglog(ks[1:end-1], exp.(F_fit.(log.(ks[1:end-1]))), label="fit q = $q_fit")
xlabel("comoving momentum |k|")
ylabel("F(k)")
title("instantaneous emission spectrum")
legend()
savefig("instant_emission_spectrum.pdf")

# string length
println("string length")
data = readdlm("string_length.dat")
taus, zeta = data[:, 1], data[:, 2]
logs = AxionStrings.tau_to_log.(taus)

figure()
plot(logs, zeta)
xlabel(raw"$\log(m_r / H)$")
ylabel(raw"$\zeta = a l / a^3 L^3 \times t^2$")
savefig("string_length.pdf")

# string movie
tmpdir = tempname()
mkdir(tmpdir)
files = String[]
strings = JSON.parse(read("strings.json", String))

println("plotting all frames")
figure()
for (i, (tau, any_strs)) in enumerate(strings)
    local strs = convert(Vector{Vector{SVector{3, Float64}}}, any_strs)
    println("$i of $(length(strings))")
    clf()
    AxionStrings.plot_strings(p, strs; colors_different=false)
    title(raw"$\tau =$" * (@sprintf "%.2f" tau) * raw", $\log(m_r/H) = $" *
          (@sprintf "%.2f" AxionStrings.tau_to_log(tau)))
    fname = joinpath(tmpdir, "strings_step=$i.jpg")
    savefig(fname)
    push!(files, fname)
end

println("creating gif")
outfile = "strings.gif"
run(Cmd(vcat(["convert", "-delay", "20", "-loop", "0"], files, outfile)))
println("done")

plt.close("all")
ion()

