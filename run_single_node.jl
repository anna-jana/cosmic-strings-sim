include("AxionStrings.jl")

using DelimitedFiles
using Printf
using JSON

p = AxionStrings.Parameter(2.0, 3.0, 1e-2, 42, 1.0, 20, 1)
s = AxionStrings.SingleNodeState(p)

# strs, mean_v, mean_v2, mean_gamma = AxionStrings.detect_strings(s, p)
# k, P1, P2 = AxionStrings.compute_spectrum_ppse(p, s, strs)
# plot(k, P1; label="corrected")
# plot(k, P2; label="uncorrected")
# legend()

write("parameter.json", json(p))

string_lengths = []
energies = []
strings = []
velocities = []

ntimes = 50
every = div(p.nsteps, ntimes)

do_spectra = true
save_strings = true

for i in 1:p.nsteps
    println("$i of $(p.nsteps)")
    if i % every == 0 || i == p.nsteps
        println("computing strings and energies...")
        strs, mean_v, mean_v2, mean_gamma = AxionStrings.detect_strings(s, p)
        push!(velocities, (s.tau, mean_v, mean_v2, mean_gamma))
        if save_strings
            push!(strings, (s.tau, strs))
        end
        push!(string_lengths, (s.tau, AxionStrings.total_string_length(s, p, strs)))
        push!(energies, (s.tau, AxionStrings.compute_energy(s, p)...))
        println("done")
    end
    if i == p.nsteps - 1 && do_spectra
        println("computing spectrum 1")
        strs, mean_v, mean_v2, mean_gamma = AxionStrings.detect_strings(s, p)
        @time wavenumber, power_ppse, power_uncorrected = AxionStrings.compute_spectrum_ppse(p, s, strs)
        _, power_screened = AxionStrings.compute_spectrum_autoscreen(p, s)
        writedlm("spectrum1.dat", hcat(wavenumber, power_ppse, power_uncorrected, power_screened))
        println("done")
    end
    AxionStrings.make_step!(s, p)
end

println("writing data files")
writedlm("string_length.dat", string_lengths)
writedlm("energies.dat", energies)
writedlm("velocities.dat", velocities)
if save_strings
    write("strings.json", json(strings))
end
println("done")

if do_spectra
    println("computing spectrum 2")
    strs, mean_v, mean_v2, mean_gamma = AxionStrings.detect_strings(s, p)
    @time wavenumber, power_ppse, power_uncorrected = AxionStrings.compute_spectrum_ppse(p, s, strs)
    _, power_screened = AxionStrings.compute_spectrum_autoscreen(p, s)
    writedlm("spectrum2.dat", hcat(wavenumber, power_ppse, power_uncorrected, power_screened))
    println("done")
end
