############## prototype implementation for string detection ############
import numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import label
from collections import defaultdict

field = np.loadtxt("final_field.dat", dtype="complex")
N = int(np.round(np.cbrt(field.size)))
field = field.reshape(N,N,N)
L = 1.0
dx = L / N

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

def find_string_points(phi):
    xs = np.linspace(0, L-dx, N)
    xx, yy, zz = np.meshgrid(xs, xs, xs)
    xy = loop_contains_string(phi, np.roll(phi, -1, 0),
        np.roll(np.roll(phi, -1, 0), -1, 1), np.roll(phi, -1, 1))
    yz = loop_contains_string(phi, np.roll(phi, -1, 1),
        np.roll(np.roll(phi, -1, 1), -1, 2), np.roll(phi, -1, 2))
    zx = loop_contains_string(phi, np.roll(phi, -1, 2),
        np.roll(np.roll(phi, -1, 2), -1, 0), np.roll(phi, -1, 0))
    coord = np.where(xy)
    x_xy, y_xy, z_xy = xx[coord] + dx/2, yy[coord] + dx/2, zz[coord]
    coord = np.where(yz)
    x_yz, y_yz, z_yz = xx[coord], yy[coord] + dx/2, zz[coord] + dx/2
    coord = np.where(zx)
    x_zx, y_zx, z_zx = xx[coord] + dx/2, yy[coord], zz[coord] + dx/2
    x = np.hstack([x_xy, x_yz, x_zx])
    y = np.hstack([y_xy, y_yz, y_zx])
    z = np.hstack([z_xy, z_yz, z_zx])
    return set(zip(x, y, z))

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
                    if dist_beginning < maximal_distance:
                        current_string.append(current_string[0])
                break
            min_d = np.inf
            min_p = None
            for p in patch:
                if len(current_string) >= 2 and p == current_string[-2]:
                    continue
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

def plot(strings, size):
    max_dist = 3*(size / 4)**2
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    for string in strings:
        x, y, z = np.array(string).T
        last = 0
        color = None
        for i in range(1, len(x)):
            d = (x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2 + (z[i] - z[i - 1])**2
            if d > max_dist or i == len(x)-1:
                l, = plt.plot(x[last:i], y[last:i], z[last:i], color=color)
                color = l.get_color()
                last = i
    plt.show()

if __name__ == "__main__":
    is_close = is_string_at(field)
    ix, iy, iz = np.where(is_close)
    patch = set(zip(ix, iy, iz))
    # we can also do this with the actual coordinates (everything times dx)
    # but this requiures passing min_string_len = 3 bc of rounding issues (I think)
    strings = nearest_neighbor_strings(patch, 3*(2)**2, N)
    plot(strings, N)
