import numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import label

# TODO: we need to pass parameters between c code and python scripts
L = 1.0

theta = np.loadtxt("axion.dat")
N = int(np.round(np.cbrt(theta.size)))
theta = theta.reshape(N, N, N)
dx = L / N
is_close = np.loadtxt("is_close.dat")
is_close = is_close.reshape(N,N,N)
patches = np.loadtxt("patches.dat")
patches = patches.reshape(N,N,N)
strings = np.loadtxt("strings.dat")
x, y, z, p, q = strings.T
p, q = p.astype("int"), q.astype("int")
q -= 1
p -= 1

def plot_is_close():
    cx, cy, cz = np.where(is_close)
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.scatter(cx, cy, cz)
    plt.show()

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
    for i in range(len(p)):
        ax.plot([x[i], x[q[i]]], [y[i], y[q[i]]], [z[i], z[q[i]]], color="black")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

def detect_strings():
    is_close = (
        (np.abs(theta - np.roll(theta, 1, 0)) > np.pi/2) |
        (np.abs(theta - np.roll(theta, -1, 0)) > np.pi/2) |
        (np.abs(theta - np.roll(theta, 1, 1)) > np.pi/2) |
        (np.abs(theta - np.roll(theta, -1, 1)) > np.pi/2) |
        (np.abs(theta - np.roll(theta, 1, 2)) > np.pi/2) |
        (np.abs(theta - np.roll(theta, -1, 2)) > np.pi/2)
    )

    labeled = np.empty((N, N, N), dtype="int")
    for i in range(N):
        sliced = is_close[i]
        embedded = np.block([
            [sliced, sliced, sliced],
            [sliced, sliced, sliced],
            [sliced, sliced, sliced],
        ])
        l, _ = label(embedded)
        labeled[i] = l[N:2*N, N:2*N]

    connections = []
    positions = {}
    for i in range(N):
        labels = np.unique(labeled[i])
        labels.sort()
        if labels[0] == 0: labels = labels[1:] # drop 0
        for l in labels:
            group1 = (i, l)
            mask = labeled[i] == l
            # calculate center
            x, y = np.where(mask)
            pos = (x.mean(), y.mean(), i)
            positions[group1] = pos
            if i == 0: continue
            # check for connections
            masked = labeled[(i - 1) % N][mask]
            connected = np.unique(masked)
            connected.sort()
            if connected[0] == 0: connected = connected[1:]
            for l2 in connected:
                group2 = ((i - 1) % N, l2)
                connections.append((group1, group2))

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    for g1, g2 in connections:
        x1, y1, z1 = positions[g1]
        x2, y2, z2 = positions[g2]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color="black")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

