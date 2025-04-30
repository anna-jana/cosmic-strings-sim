import json
import numpy as np
import AxionStrings
import dataclasses

p = AxionStrings.make_parameter(2.0, 3.0, 1e-2, 42, 1.0, 20, 1)
s = AxionStrings.make_state(p)

#ks_init, P_init = AxionStrings.compute_spectrum_autoscreen(p, s)

if s.rank == s.root:
    with open("parameter.json", "w") as f:
        json.dump(dataclasses.asdict(p), f)

energy_data = []
velocity_data = []
string_data = []

for _ in range(p.nsteps):
    AxionStrings.step(s, p)

    #strs, mean_v, mean_v2, mean_gamma = AxionStrings.detect_strings(s, p)
    #l = AxionStrings.total_string_length(s, p, strs)
    #e = AxionStrings.compute_energy(s, p)

    #if s.rank == s.root:
    #    velocity_data.append((s.tau, mean_v, mean_v2, mean_gamma))
    #    string_data.append((s.tau, l))
    #    energy_data.append((s.tau,) + e)

# ks, P = AxionStrings.compute_spectrum_autoscreen(p, s)

#if s.rank == s.root:
#    np.savetxt("energies.dat", energy_data)
#    np.savetxt("spectrum1.dat", ks_init, P_init)
#    np.savetxt("spectrum2.dat", ks, P)
#    np.savetxt("string_length.dat", string_data)
#    np.savetxt("velocities.dat", velocity_data)

AxionStrings.finish_mpi()

