import json, sys, itertools, os
import numpy as np
import cosmology

if len(sys.argv) == 1:
    for i in itertools.count(1):
        dirname = f"run{i}_output"
        if os.path.exists(dirname):
            break
elif len(sys.argv) == 2:
    dirname = sys.argv[1]
else:
    raise ValueError("invalid argument")

def create_output_path(fname):
    return os.path.join(dirname, fname)

with open(create_output_path("parameter.json"), "r") as f:
    parameter = json.load(f)

log_start = parameter["LOG_START"]
log_end = parameter["LOG_END"]
L = parameter["L"]
N = parameter["N"]
dtau = parameter["DELTA"]

tau_start = cosmology.log_to_tau(log_start)
tau_end = cosmology.log_to_tau(log_end)
dx = L / N

fname = create_output_path("final_field.dat")
final_field = np.loadtxt(fname, dtype="complex")
final_field = final_field.reshape(N, N, N)

string_step, string_id, string_x, string_y, string_z = np.loadtxt(create_output_path("strings.dat")).T

energy_step, axion_kinetic, axion_gradient, axion_total, radial_kinetic, \
    radial_gradient, radial_potential, radial_total, interaction, total = \
    np.loadtxt(create_output_path("energies.dat")).T

