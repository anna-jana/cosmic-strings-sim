import numpy as np
import matplotlib.pyplot as plt
import numpy

x, cos, sin = np.loadtxt("fftw_test.dat").T
plt.plot(x, cos, label="f = cos")
plt.plot(x, sin, label="d f / d x should be - sin x")
plt.legend()
plt.show()

