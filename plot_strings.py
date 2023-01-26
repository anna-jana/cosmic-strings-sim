import numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import label
from collections import defaultdict

# util functions
def plot_boxes(boxes):
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.voxels(boxes)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

# TODO: we need to pass parameters between c code and python scripts

#################### load all data output from c code #####################
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

############################## plot the data ###########################
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
