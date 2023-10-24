include("AxionStrings.jl")
include("util.jl")

using PyPlot
using DelimitedFiles
using Printf

# s, p = AxionStrings.init()
# AxionStrings.run_simulation!(s, p)
# strs = AxionStrings.detect_strings(s, p)

s, p = AxionStrings.init()
tmpdir = tempname()
mkdir(tmpdir)
files = String[]
string_lengths = Float64[]
taus = Float64[]
ioff()
AxionStrings.run_simulation!(s, p, 50) do
    println("plotting... ")
    strs = AxionStrings.detect_strings(s, p)
    push!(string_lengths, AxionStrings.total_string_length(p, strs))
    push!(taus, s.tau)
    figure()
    plot_strings(p, strs; colors_different=false)
    title(raw"$\tau =$" * (@sprintf "%.2f" s.tau) * raw", $\log(m_r/H) = $" * (@sprintf "%.2f" AxionStrings.tau_to_log(s.tau)))
    fname = joinpath(tmpdir, "strings_step=$(s.step).jpg")
    savefig(fname)
    push!(files, fname)
    println("done")
end
println("writing strings lengths")
writedlm("strings_length.csv", hcat(taus, string_lengths))
println("done")
println("creating gif")
outfile = "strings.gif"
run(Cmd(vcat(["convert", "-delay", "20", "-loop", "0"],
         files,
         outfile)))
println("done")
ion()


# spec = wavenumber, power = AxionStrings.compute_spectrum(p, s, strs)
# plot(wavenumber, power)
# save_spectrum(spec, "mc_spec.hdf5")
