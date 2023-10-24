using PyPlot
using HDF5
using StaticArrays
using LinearAlgebra

include("AxionStrings.jl")

function plot_energies(energies)
    (axion_kinetic, axion_gradient, axion_total,
            radial_kinetic, radial_gradient, radial_potential, radial_total,
            interaction, total) = [[e[i] for e in energies] for i in 1:length(energies[1])]
    figure()
    plot(log, axion_kinetic, color="tab:blue", ls="-", label="axion, kinetic")
    plot(log, axion_gradient, color="tab:blue", ls="--", label="axion, gradient")
    plot(log, axion_total, color="tab:blue", ls="-", lw=2, label="axion")
    plot(log, radial_kinetic, color="tab:orange", ls="-", label="radial, kinetic")
    plot(log, radial_gradient, color="tab:orange", ls="--", label="radial, gradient")
    plot(log, radial_potential, color="tab:orange", ls=":", label="radial, potential")
    plot(log, radial_total, color="tab:orange", ls="-", lw=2, label="radial")
    plot(log, interaction, color="tab:green", ls="-", label="interaction axion and radial = strings")
    plot(log, total, color="black", ls="-", lw=2, label="total energy density")
    yscale("log")
    xlabel(r"$log(m_r / H)$")
    ax = gca()
    ax.invert_xaxis()
    ylabel(r"averaged energy density $f_a^2 m_r^2$\n")
    legend()
end

function plot_spectrum(p :: AxionStrings.Parameter, bins, P_ppse)
    figure()
    step(bins, P_ppse, where="mid", label="corrected spectrum of free axions")
    xlabel("k")
    ylabel("P(k)")
    xscale("log")
    yscale("log")
    title("log = $(p.log_end)")
    legend()
    show()
end

function save_spectrum(spec, filename)
    wavenumber, power = spec
    h5open(filename, "w") do f
        g = create_group(f, "spectrum")
        g["wavenumber"] = collect(wavenumber)
        g["power"] = power
    end
    return nothing
end

function load_spectrum(filename)
    wavenumber, power = h5open(filename, "r") do f
        g = read(f, "spectrum")
        return g["wavenumber"], g["power"]
    end
    return wavenumber, power
end


function plot_strings(params :: AxionStrings.Parameter, strings :: Vector{Vector{SVector{3, Float64}}}; colors_different=false)
    fig = figure()
    fig.add_subplot(projection="3d")

    color = nothing

    for string in strings
        xs = [string[1][1]]
        ys = [string[1][2]]
        zs = [string[1][3]]
        prev = string[1]
        if colors_different
            color = nothing
        end

        for p in string[2:end]
            if norm(p .- prev) <= sqrt(3)
                push!(xs, p[1])
                push!(ys, p[2])
                push!(zs, p[3])
            else
                l, = plot(xs, ys, zs, color=color)
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

        plot(xs, ys, zs, color=color)
    end

    xlabel("x")
    ylabel("y")
    zlabel("z")

    return nothing
end

function plot_required_size()
    log_end = range(1.0, 9.0, 100)
    required_N = [AxionStrings.sim_params_from_physical_scale(l)[2] for l in log_end]
    required_bytes = @. 4 * 4 * required_N^3 / (1024^3)
    plot(log_end, required_bytes, label="required by simulation")
    axhline(8; color="k", ls="--", label="laptop")
    xlabel("scale log(m_r / H)")
    ylabel("giga bytes")
    yscale("log")
    legend()
    show()
end
