import json
import dataclasses
import numpy as np
import AxionStrings
import energy
import strings
import spectrum

p = AxionStrings.make_parameter(2.0, 3.0, 1e-3, 42, 1.0, 20, 1)
s = AxionStrings.State(p)

ks_init, P_init = spectrum.compute_spectrum_autoscreen(s)

if s.rank == s.root:
    with open("parameter.json", "w") as f:
        json.dump(dataclasses.asdict(p), f)

energy_data = []
velocity_data = []
string_data = []

for _ in range(p.nsteps):
    s.do_step()

    # analysis
    e = energy.compute_energy(s, p)
    string_points, velocities = strings.find_string_cores(s)
    l, induvidual_strings = strings.find_induvidual_strings(s, string_points)
    if s.rank == s.root:
        energy_data.append((s.tau,) + e)
        velocity_data.append((s.tau,) + velocities)
        string_data.append((s.tau, l))

ks, P = spectrum.compute_spectrum_autoscreen(s)

if s.rank == s.root:
    np.savetxt("energies.dat", energy_data)
    np.savetxt("velocities.dat", velocity_data)
    np.savetxt("string_length.dat", string_data)
    np.savetxt("spectrum1.dat", ks_init, P_init)
    np.savetxt("spectrum2.dat", ks, P)

s.finish_mpi()

