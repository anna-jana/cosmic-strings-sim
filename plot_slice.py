import numpy as np, matplotlib.pyplot as plt
import sys
fname = sys.argv[1] if len(sys.argv) > 1 else "slice.dat"
s = np.loadtxt(fname, dtype="complex")
theta = np.angle(s)
x = np.linspace(0, 1, s.shape[0])
y = np.linspace(0, 1, s.shape[1])
plt.pcolormesh(x, y, theta, cmap="twilight")
plt.colorbar(label=r"$\theta = \arg(\phi(x, y))$")
plt.xlabel("x")
plt.ylabel("y")
plt.title("plot of 2D slice of 3D string simulation")
plt.show()
