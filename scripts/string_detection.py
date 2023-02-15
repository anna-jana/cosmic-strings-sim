############## prototype implementation for string detection ############
import sys
from collections import defaultdict
import numpy as np, matplotlib.pyplot as plt
import load_data

# (string contention method from Moore at al.)
def crosses_real_axis(phi1, phi2):
    return np.imag(phi1) * np.imag(phi2) < 0

def handedness(phi1, phi2):
    return np.sign(np.imag(phi1 * np.conj(phi2)))

def loop_contains_string(phi1, phi2, phi3, phi4):
    loop = (
          crosses_real_axis(phi1, phi2) * handedness(phi1, phi2)
        + crosses_real_axis(phi2, phi3) * handedness(phi2, phi3)
        + crosses_real_axis(phi3, phi4) * handedness(phi3, phi4)
        + crosses_real_axis(phi4, phi1) * handedness(phi4, phi1)
    )
    return np.abs(loop) == 2

def is_string_at(phi):
    xy = loop_contains_string(phi, np.roll(phi, -1, 0),
        np.roll(np.roll(phi, -1, 0), -1, 1), np.roll(phi, -1, 1))
    yz = loop_contains_string(phi, np.roll(phi, -1, 1),
        np.roll(np.roll(phi, -1, 1), -1, 2), np.roll(phi, -1, 2))
    zx = loop_contains_string(phi, np.roll(phi, -1, 2),
        np.roll(np.roll(phi, -1, 2), -1, 0), np.roll(phi, -1, 0))
    return xy | yz | zx

def cyclic_dist_squared_1d(x1, x2, D):
    return min((x1 - x2)**2, (D - x1 + x2)**2, (D - x2 + x1)**2)

def cyclic_dist_squared(p1, p2, D):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return (
        cyclic_dist_squared_1d(x1, x2, D) +
        cyclic_dist_squared_1d(y1, y2, D) +
        cyclic_dist_squared_1d(z1, z2, D)
    )

def nearest_neighbor_strings(patch, maximal_distance, side_length, min_string_len = 3):
    patch = patch.copy()
    strings = []
    while patch:
        current_string = [patch.pop()]
        while True:
            if not patch:
                if len(current_string) >= 2:
                    dist_beginning = cyclic_dist_squared(
                            current_string[-1], current_string[0], side_length)
                    if dist_beginning <= maximal_distance:
                        current_string.append(current_string[0])
                break
            min_d = np.inf
            min_p = None
            for p in patch:
                d = cyclic_dist_squared(p, current_string[-1], side_length)
                if d < min_d:
                    min_d = d
                    min_p = p
            if len(current_string) >= min_string_len:
                dist_beginning = cyclic_dist_squared(
                        current_string[-1], current_string[0], side_length)
                if dist_beginning < min_d:
                    break
            if min_d > maximal_distance:
                break
            patch.remove(min_p)
            current_string.append(min_p)
        strings.append(current_string)
    return strings

def plot(strings, size, scale, data=None, step=None):
    max_dist = 3*(scale * size / 4)**2
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(0, size*scale)
    ax.set_ylim(0, size*scale)
    ax.set_zlim(0, size*scale)
    if step is not None:
        ax.set_title(f"$\\tau = {data.tau_start + step*data.dtau}$")
    for string in strings:
        x, y, z = np.array(string).T * scale
        last = 0
        color = None
        for i in range(1, len(x)):
            d = (x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2 + (z[i] - z[i - 1])**2
            if d > max_dist or i == len(x)-1:
                l, = plt.plot(x[last:i], y[last:i], z[last:i], color=color)
                color = l.get_color()
                last = i

if __name__ == "__main__":
    data = load_data.OutputDir(sys.argv[1])
    is_close = is_string_at(data.final_field)
    ix, iy, iz = np.where(is_close)
    patch = set(zip(ix, iy, iz))
    # we can also do this with the actual coordinates (everything times dx)
    # but this requiures passing min_string_len = 3 bc of rounding issues (I think)
    strings = nearest_neighbor_strings(patch, 3*(2)**2, data.N)
    plot(strings, data.N, data.dx)
    plt.show()
