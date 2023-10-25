include("AxionStrings.jl")

using PyPlot
using DelimitedFiles
using Printf

s, p = AxionStrings.init()

tmpdir = tempname()
mkdir(tmpdir)
files = String[]
ioff()

string_lengths = []
energies = []

nframes = 50

AxionStrings.run_simulation!(s, p, nframes) do
    println("plotting... ")
    strs = AxionStrings.detect_strings(s, p)
    push!(string_lengths, (s.tau, AxionStrings.total_string_length(s, p, strs)))
    push!(energies, (s.tau, AxionStrings.compute_energy(s, p)...))
    figure()
    AxionStrings.plot_strings(p, strs; colors_different=false)
    title(raw"$\tau =$" * (@sprintf "%.2f" s.tau) * raw", $\log(m_r/H) = $" * (@sprintf "%.2f" AxionStrings.tau_to_log(s.tau)))
    fname = joinpath(tmpdir, "strings_step=$(s.step).jpg")
    savefig(fname)
    push!(files, fname)
    println("done")
end

println("writing data files")
writedlm("string_length.dat", string_lengths)
writedlm("energies.dat", energies)
println("done")

println("creating gif")
outfile = "strings.gif"
run(Cmd(vcat(["convert", "-delay", "20", "-loop", "0"], files, outfile)))
println("done")
plt.close("all")
ion()


# spec = wavenumber, power = AxionStrings.compute_spectrum(p, s, strs)
# plot(wavenumber, power)
# save_spectrum(spec, "mc_spec.hdf5")
