# TODO: make this use the output correctly!!!

import numpy as np, matplotlib.pyplot as plt
import sys, os.path
import load_data

data = load_data.OutputDir(sys.argv[1])
spectrum_step = data.spectrum_step.max()

pybins, pyspectrum_uncorrected, pyspectrum = np.loadtxt(os.path.join(os.path.dirname(__file__), "pyspectrum.dat"), unpack=True)

bins = data.bins[data.spectrum_step == spectrum_step][1:]
plt.step(bins, data.spectrum[data.spectrum_step == spectrum_step][1:],
        where="pre", label="corrected spectrum of free axions")
plt.step(bins, data.spectrum_uncorrected[data.spectrum_step == spectrum_step][1:],
        where="pre", label="uncorrected spectrum of free axions")

plt.step(pybins[1:], pyspectrum[1:],
        where="pre", label="python corrected spectrum of free axions")
plt.step(pybins[1:], pyspectrum_uncorrected[1:],
        where="pre", label="python uncorrected spectrum of free axions")
plt.xlabel("k")
plt.ylabel("P(k)")
plt.xscale("log")
plt.yscale("log")
plt.title(f"log = {data.log_end:.2f}")
plt.legend()
plt.show()
