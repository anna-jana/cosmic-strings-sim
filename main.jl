# mpirun -n 4 julia --project=. run_mpi.jl

include("AxionStrings.jl")

using JSON
using DelimitedFiles

p = AxionStrings.Parameter(2.0, 3.0, 1e-2, 42, 1.0, 20, 1)
s = AxionStrings.State(p)

ks_init, P_init = AxionStrings.compute_spectrum_autoscreen(p, s)

if s.rank == s.root
    write("parameter.json", json(p))
end

energy_data = []
velocity_data = []
string_data = []

for _ in 1:p.nsteps
    AxionStrings.step!(s, p)
    strs, mean_v, mean_v2, mean_gamma = AxionStrings.detect_strings(s, p)
    l = AxionStrings.total_string_length(s, p, strs)
    e = AxionStrings.compute_energy(s, p)
    if s.rank == s.root
        push!(velocity_data, (s.tau, mean_v, mean_v2, mean_gamma))
        push!(string_data, (s.tau, l))
        push!(energy_data, (s.tau, e...))
    end
end

ks, P = AxionStrings.compute_spectrum_autoscreen(p, s)

if s.rank == s.root
    writedlm("energies.dat", energy_data)
    writedlm("spectrum1.dat", ks_init, P_init)
    writedlm("spectrum2.dat", ks, P)
    writedlm("string_length.dat", string_data)
    writedlm("velocities.dat", velocity_data)
end

AxionStrings.finish_mpi!(s)

