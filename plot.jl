println("loading packages")

include("AxionStrings.jl")

using PyPlot
using DelimitedFiles
using Interpolations
using JSON
using Polynomials
using Printf
using StaticArrays
using LinearAlgebra

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

plot(logs, axion_kinetic, color="tab:blue", ls=":", label="axion, kinetic")
plot(logs, axion_gradient, color="tab:blue", ls="--", label="axion, gradient")
plot(logs, axion_total, color="tab:blue", ls="-", label="axion")
plot(logs, radial_kinetic, color="tab:orange", ls=":", label="radial, kinetic")
plot(logs, radial_gradient, color="tab:orange", ls="--", label="radial, gradient")
plot(logs, radial_potential, color="tab:orange", ls="-.", label="radial, potential")
plot(logs, radial_total, color="tab:orange", ls="-", label="radial")
plot(logs, interaction, color="tab:green", ls="-", label="interaction axion and radial = strings")
plot(logs, total, color="black", ls="--", label="total energy density")


# plot(logs, axion_kinetic ./ total, color="tab:blue", ls="-", label="axion, kinetic")
# plot(logs, axion_gradient ./ total, color="tab:blue", ls="--", label="axion, gradient")
# plot(logs, axion_total ./ total, color="tab:blue", ls="-", lw=2, label="axion")
# plot(logs, radial_kinetic ./ total, color="tab:orange", ls="-", label="radial, kinetic")
# plot(logs, radial_gradient ./ total, color="tab:orange", ls="--", label="radial, gradient")
# plot(logs, radial_potential ./ total, color="tab:orange", ls=":", label="radial, potential")
# plot(logs, radial_total ./ total, color="tab:orange", ls="-", lw=2, label="radial")
# plot(logs, interaction ./ total, color="tab:green", ls="-", label="interaction axion and radial = strings")

yscale("log")
xlabel(raw"$log(m_r / H)$")
ylabel(raw"averaged energy density $f_a^2 m_r^2$\n")
legend()
savefig("energy_densities.pdf")

# spectra
println("spectra")
data = readdlm("spectrum1.dat")
k1, P1_ppse, P1_screened = data[:, 1], data[:, 2], data[:, 3]
data = readdlm("spectrum2.dat")
k2, P2_ppse, P2_screened = data[:, 1], data[:, 2], data[:, 3]
tau1 = p.Delta_tau * (p.nsteps - 1)
tau2 = p.Delta_tau * p.nsteps

log1 = AxionStrings.tau_to_log(tau1)
log2 = AxionStrings.tau_to_log(tau2)

log_mid, ks, F_ppse = AxionStrings.compute_instanteous_emission_spectrum(P1_ppse, P2_ppse, k1, k2, tau1, tau2)
_, _, F_screened = AxionStrings.compute_instanteous_emission_spectrum(P1_screened, P2_screened, k1, k2, tau1, tau2)

#F_fit_ppse = fit(log.(ks[1:end-1]), log.(F_ppse[1:end-1]), 1)
#F_fit_screened = fit(log.(ks[1:end-1]), log.(F_screened[1:end-1]), 1)
#q_fit_ppse = -F_fit_ppse[1]
#q_fit_screened = -F_fit_screened[1]

figure()
plot(k1, P1_ppse, label="ppse, log = $log1")
plot(k2, P2_ppse, label="ppse, log = $log2")
plot(k1, P1_screened, label="screened, log = $log1")
plot(k2, P2_screened, label="screeend, log = $log2")
xlabel("comoving momentum |k|")
ylabel("power spectrum P(k)")
xscale("log")
yscale("log")
legend()
title("spectrum of free axions")
savefig("spectra.pdf")

figure()
loglog(ks, F_ppse, label="ppse, simulation at log=$log_mid")
loglog(ks, F_screened, label="screened, simulation at log=$log_mid")
#loglog(ks[1:end-1], exp.(F_fit_ppse.(log.(ks[1:end-1]))), label="ppse, fit q = $q_fit_ppse")
#loglog(ks[1:end-1], exp.(F_fit_screened.(log.(ks[1:end-1]))), label="screened, fit q = $q_fit_screened")
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
function plot_strings(params :: AxionStrings.Parameter, strings; colors_different=false)
    fig = gcf()
    fig.add_subplot(projection="3d")

    for string in strings
        xs = [string[1][1]]
        ys = [string[1][2]]
        zs = [string[1][3]]
        prev = string[1]
        color = nothing

        for p in string[2:end]
            if norm(p .- prev) <= sqrt(3)
                push!(xs, p[1])
                push!(ys, p[2])
                push!(zs, p[3])
            else
                l, = plot(xs .* params.dx, ys .* params.dx, zs .* params.dx, color=colors_different ? color : "tab:blue")
                color = l.get_color()
                xs = [p[1]]
                ys = [p[2]]
                zs = [p[3]]
            end
            prev = p
        end

        if norm(string[1] - string[end]) <= sqrt(3)
            push!(xs, string[1][1])
            push!(ys, string[1][2])
            push!(zs, string[1][3])
        end

        plot(xs .* params.dx, ys .* params.dx, zs .* params.dx, color=colors_different ? color : "tab:blue")
    end

    xlabel(raw"$x m_r$")
    ylabel(raw"$y m_r$")
    zlabel(raw"$z m_r$")

    return nothing
end

function make_string_movie(p::AxionStrings.Parameter)
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
        plot_strings(p, strs; colors_different=false)
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
end

function field_plot(p, s)
    slice = 1
    a = AxionStrings.tau_to_a(s.tau)
    r = AxionStrings.compute_radial_mode.(s.psi, a)
    theta = angle.(s.psi)
    xs = range(0, p.L, p.N)
    kappa = AxionStrings.tau_to_log(s.tau)

    figure()

    subplot(2, 1, 1)
    pcolormesh(xs, xs, r[slice, :, :])
    xlabel("x_comoving m_r")
    ylabel("y_comoving m_r")
    colorbar(label="radial mode / f_a")

    subplot(2, 1, 2)
    pcolormesh(xs, xs, theta[slice, :, :], cmap="twilight")
    xlabel("x_comoving m_r")
    ylabel("y_comoving m_r")
    colorbar(label="theta")

    title("log = $kappa")
end

plt.close("all")
ion()
