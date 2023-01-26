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


def find_strings_3d_patches(patches):
    N = patches.shape[0]
    remaining_idx = {(x, y, z) for x, y, z in zip(*np.where(patches))}
    patches = []
    while remaining_idx:
        current_patch = set([])
        not_expanded = set([remaining_idx.pop()])
        while not_expanded:
            x, y, z = not_expanded.pop()
            current_patch.add((x, y, z))
            for dx in -1,0,1:
                for dy in -1,0,1:
                    for dz in -1,0,1:
                        if not (dx == 0 and dy == 0 and dz == 0):
                            neighbor_p = (x + dx) % N, (y + dy) % N, (z + dz) % N
                            if neighbor_p in remaining_idx:
                                not_expanded.add(neighbor_p)
                                remaining_idx.remove(neighbor_p)
        patches.append(current_patch)
    return patches

def cyclic_dist_squared_1d(x1, x2, L):
    return min((x1 - x2)**2, (L - x1 + x2)**2, (L - x2 + x1)**2)

def cyclic_dist_squared(p1, p2, L):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return (
        cyclic_dist_squared_1d(x1, x2, L) +
        cyclic_dist_squared_1d(y1, y2, L) +
        cyclic_dist_squared_1d(z1, z2, L)
    )

def nearest_neighbor_strings(patch):
    patch = patch.copy()
    strings = []
    while patch:
        current_string = [patch.pop()]
        while True:
            if not patch:
                print("open string")
                break
            min_d = np.inf
            min_p = None
            for p in patch:
                if p == current_string[-1]:
                    continue
                if len(current_string) >= 2 and p == current_string[-2]:
                    continue
                d = cyclic_dist_squared(p, current_string[-1], L)
                if d < min_d:
                    min_d = d
                    min_p = p
            dist_beginning = cyclic_dist_squared(current_string[-1], current_string[0], L)
            print(dist_beginning, min_d)
            if dist_beginning < min_d:
                print("loop", len(current_string))
                break
            patch.remove(min_p)
            current_string.append(min_p)
        strings.append(current_string)
    return strings

is_close = is_string_at(field)
patches = find_strings_3d_patches(is_close)
fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
for patch in patches:
    strings = nearest_neighbor_strings(patch)
    print("****************************")
    for string in strings:
        x, y, z = np.array(string).T
        ax.scatter(x, y, z)
plt.show()


