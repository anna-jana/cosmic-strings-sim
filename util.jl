# function plot_strings(params :: AxionStrings.Parameter, strings :: Vector{Vector{SVector{3, Float64}}}; colors_different=false)
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
