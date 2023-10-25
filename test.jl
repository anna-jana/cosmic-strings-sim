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

ntimes = 50
every = div(p.nsteps, ntimes)

for i in 1:p.nsteps
    println("$i of $(p.nsteps)")
    if i % every == 0 || i == p.nsteps
        println("plotting... ")
        strs = AxionStrings.detect_strings(s, p)
        push!(string_lengths, (s.tau, AxionStrings.total_string_length(s, p, strs)))
        push!(energies, (s.tau, AxionStrings.compute_energy(s, p)...))
        figure()
        AxionStrings.plot_strings(p, strs; colors_different=false)
        title(raw"$\tau =$" * (@sprintf "%.2f" s.tau) * raw", $\log(m_r/H) = $" *
              (@sprintf "%.2f" AxionStrings.tau_to_log(s.tau)))
        fname = joinpath(tmpdir, "strings_step=$(s.step).jpg")
        savefig(fname)
        push!(files, fname)
        println("done")
    end
    if i == p.nsteps - 1
        println("computing spectrum 1")
        strs = AxionStrings.detect_strings(s, p)
        @time wavenumber, power = AxionStrings.compute_spectrum(p, s, strs)
        writedlm("spectrum1.dat", hcat(wavenumber, power))
        println("done")
    end
    AxionStrings.make_step!(s, p)
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

println("computing spectrum 2")
strs = AxionStrings.detect_strings(s, p)
@time wavenumber, power = AxionStrings.compute_spectrum(p, s, strs)
writedlm("spectrum2.dat", hcat(wavenumber, power))
println("done")
# plot(wavenumber, power)
# save_spectrum(spec, "mc_spec.hdf5")
