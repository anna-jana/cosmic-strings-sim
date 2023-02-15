import numpy as np, matplotlib.pyplot as plt
import string_detection, load_data

def plot(step_used):
    string_list = []
    for i in np.unique(load_data.string_id[load_data.string_step == step_used]):
        mask = (load_data.string_step == step_used) & (load_data.string_id == i)
        string_list.append(list(zip(load_data.string_x[mask], load_data.string_y[mask], load_data.string_z[mask])))
    string_detection.plot(string_list, load_data.N, load_data.dx, step=step_used)

plot(load_data.string_step[0])
plot(load_data.string_step[-1])
plt.show()
