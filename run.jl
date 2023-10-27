include("AxionStrings.jl")

using DelimitedFiles
using Printf
using JSON

s, p = AxionStrings.init()

write("parameter.json", json(p))

string_lengths = []
energies = []
strings = []

ntimes = 50
every = div(p.nsteps, ntimes)

do_spectra = false

for i in 1:p.nsteps
    println("$i of $(p.nsteps)")
    if i % every == 0 || i == p.nsteps
        println("computing strings and energies...")
        strs = AxionStrings.detect_strings(s, p)
        push!(strings, (s.tau, strs))
        push!(string_lengths, (s.tau, AxionStrings.total_string_length(s, p, strs)))
        push!(energies, (s.tau, AxionStrings.compute_energy(s, p)...))
        println("done")
    end
    if i == p.nsteps - 1 && do_spectra
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
write("strings.json", json(strings))
println("done")

if do_spectra
    println("computing spectrum 2")
    strs = AxionStrings.detect_strings(s, p)
    @time wavenumber, power = AxionStrings.compute_spectrum(p, s, strs)
    writedlm("spectrum2.dat", hcat(wavenumber, power))
    println("done")
end

@show hash(s.phi)
