import sys
import numpy as np, matplotlib.pyplot as plt
import string_detection, load_data

data = load_data.OutputDir(sys.argv[1])

def plot(step_used):
    string_list = []
    for i in np.unique(data.string_id[data.string_step == step_used]):
        mask = (data.string_step == step_used) & (data.string_id == i)
        string_list.append(list(zip(data.string_x[mask], data.string_y[mask], data.string_z[mask])))
    string_detection.plot(string_list, data.N, data.dx, step=step_used, data=data)

plot(data.string_step[0])
plot(data.string_step[-1])
plt.show()
