import numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import label
from collections import defaultdict

# TODO: we need to pass parameters between c code and python scripts

patches = np.loadtxt("patches.dat")
N = int(np.round(np.cbrt(patches.size)))
patches = patches.reshape(N,N,N)

L = 1.0
dx = L / N

strings = np.loadtxt("strings.dat")
step, string_id, patch_id, connected, x, y, z = strings.T
patch_id, connected = patch_id.astype("int") - 1, connected.astype("int") - 1
positions = dict(zip(patch_id, zip(x, y, z)))
connections = defaultdict(lambda: set())
for p, q in zip(patch_id, connected):
    connections[p].add(q)

field = np.loadtxt("final_field.dat", dtype="complex")
field = field.reshape(N,N,N)

def plot_patches():
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    for i in np.unique(patches):
        if i != 0:
            ps = np.where(patches == i)
            ax.scatter(*ps)
    plt.show()

def plot_string_points():
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.scatter(x,y,z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

def plot_string_connections():
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    for p1 in connections:
        for p2 in connections[p1]:
            if p2 == -1:
                continue
            pos1 = positions[p1]
            pos2 = positions[p2]
            d = np.linalg.norm(np.array(pos1) - pos2)
            if d > (L/4)**2:
                continue
            ax.plot(*zip(pos1, pos2), color="black")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

def plot_compare_field_and_patches(i):
    plt.figure()
    plt.subplot(2,1,1)
    plt.pcolormesh(np.angle(field[i]), cmap="twilight")
    plt.subplot(2,1,2)
    plt.pcolormesh(patches[i])
    plt.show()

def plot_boxes(boxes):
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.voxels(boxes)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

# prototype implementation for string detection
def crosses_real_axis(phi1, phi2):
    return np.imag(phi1) * np.imag(phi2) < 0

def handedness(phi1, phi2):
    return np.sign(np.imag(phi1 * np.conj(phi2)))

def loop_contains_string(phi1, phi2, phi3, phi4):
    loop = (
        + crosses_real_axis(phi1, phi2) * handedness(phi1, phi2)
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
    labeled, nlabels = label(patches != 0, np.ones((3, 3, 3), dtype="int"))
    # connect patches on opposite sides of the grid
    # to enforce periodic boundary conditions which
    # are not respected by the scipy function labels
    labels = set(range(1, nlabels + 1))
    for left, right in [(labeled[0, :, :], labeled[-1, :, :]),
                        (labeled[:, 0, :], labeled[:, -1, :]),
                        (labeled[:, :, 0], labeled[:, :, -1])]:
        idx1, idx2 = np.where((left != right) & (left != 0) & (right != 0))
        patches_left = left[idx1, idx2]
        patches_right = right[idx1, idx2]
        for x, y in zip(patches_left, patches_right):
            labeled[labeled == x] = y
            labels.remove(x)
    return labels, labeled

labels, labeled = find_strings_3d_patches(patches)

def nearest_neighbor_strings(x, y, z):
    pass







