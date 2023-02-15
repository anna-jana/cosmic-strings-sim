import numpy as np, matplotlib.pyplot as plt
import sys
import string_detection, cosmology

assert len(sys.argv) == 2
step_used = int(sys.argv[1])

step, string_id, x, y, z = np.loadtxt("strings.dat").T

string_list = []
for i in np.unique(string_id[step == step_used]):
    mask = (step == step_used) & (string_id == i)
    string_list.append(list(zip(x[mask], y[mask], z[mask])))

string_detection.plot(string_list, cosmology.N, cosmology.dx, step=step_used)
