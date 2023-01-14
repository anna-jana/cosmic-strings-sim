import numpy as np
import matplotlib.pyplot as plt
import numpy

x, cos, deriv = np.loadtxt("fftw_test.dat").T
plt.plot(x, cos, label="f = cos")
plt.plot(x, deriv , label="d f / d x should be - sin x")
plt.legend()
plt.show()

