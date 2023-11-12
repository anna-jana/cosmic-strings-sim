include("AxionStrings.jl")

using PyPlot
using DelimitedFiles
using Interpolations
using JSON
using Polynomials
using Printf
using StaticArrays
using LinearAlgebra

function load_parameters()
    p = JSON.parse(read("parameter.json", String)) # load json
    p = Dict([Symbol(s) => v for (s, v) in p]) # string keys -> symbols
    p = AxionStrings.Parameter(; p...) # create type
    return p
end

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

function make_string_movie(p :: AxionStrings.Parameter)
    tmpdir = tempname()
    mkdir(tmpdir)
    files = String[]
    strings = JSON.parse(read("strings.json", String))
    ioff()

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

    plt.close("all")
    ion()
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