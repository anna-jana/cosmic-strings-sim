import numpy as np, matplotlib.pyplot as plt
import prototype_string_detection

strings = np.loadtxt("strings.dat")
step, string_id, x, y, z = strings.T
string_list = []
step_used = np.unique(step)[-1]
for i in np.unique(string_id[step == step_used]):
    mask = (step == step_used) & (string_id == i)
    string_list.append(list(zip(x[mask], y[mask], z[mask])))
prototype_string_detection.plot(string_list, prototype_string_detection.N)
