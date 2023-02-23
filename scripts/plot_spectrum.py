# TODO: make this use the output correctly!!!

import numpy as np, matplotlib.pyplot as plt

import load_data

data = load_data.OutputDir("run1_output")

spectrum_step, bins, spectrum_uncorrected, spectrum = np.loadtxt("run1_output/spectrum.dat", unpack=True)

plt.step(bins[1:], spectrum[1:], where="pre", label="corrected spectrum of free axions")
plt.step(bins[1:], spectrum_uncorrected[1:], where="pre", label="uncorrected spectrum of free axions")
plt.xlabel("k")
plt.ylabel("P(k)")
plt.xscale("log")
plt.yscale("log")
plt.title(f"log = {data.log_end:.2f}")
plt.legend()
plt.show()
