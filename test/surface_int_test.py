import numpy as np
import matplotlib.pyplot as plt

plt.ion()

r_max = 2.0
r_min = 0.0
xs = np.linspace(-r_max, r_max, 100)
Delta_x = xs[1] - xs[0]
x, y, z = np.meshgrid(xs, xs, xs)
r2 = x**2 + y**2 + z**2
r2 = r2.ravel()
f = np.sqrt(r2).ravel()


nbins = 20
bin_width = (r_max - r_min) / nbins
bin_index = np.floor(np.sqrt(r2) / bin_width).astype("int")

integrant = f

integrals = []
for i in range(nbins):
    vol = 4/3*np.pi * (((i + 1)*bin_width)**3 - (i*bin_width)**3)
    area = 4*np.pi * (i*bin_width + bin_width/2)**2
    print(area/vol, 1/bin_width)
    integrals.append(Delta_x**3 * np.sum(integrant[i == bin_index]) / vol * area)

rs = np.arange(nbins) * bin_width + bin_width/2
# analytically: f = r* on r = r* ===> integral d Omega f = 4\pi*r**2 * r
analytic = rs**2 * 4*np.pi * rs
plt.plot(rs, analytic)
plt.plot(rs, integrals, "x")
#plt.xscale("log")
#plt.yscale("log")

