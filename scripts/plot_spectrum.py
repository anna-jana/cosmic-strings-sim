# TODO: make this use the output correctly!!!

import numpy as np, matplotlib.pyplot as plt
import sys, os.path
import load_data, spectrum

data = load_data.OutputDir(sys.argv[1])
spectrum_step = data.spectrum_step.max()

bins = data.bins[data.spectrum_step == spectrum_step][1:]
plt.step(bins, data.spectrum[data.spectrum_step == spectrum_step][1:],
        where="mid", label="corrected spectrum of free axions")
plt.step(bins, data.spectrum_uncorrected[data.spectrum_step == spectrum_step][1:],
        where="mid", label="uncorrected spectrum of free axions")

plt.step(spectrum.bin_k[1:], spectrum.P_ppse[1:], where="mid", label="python corrected spectrum of free axions")
plt.step(spectrum.bin_k[1:], spectrum.P_full[1:], where="mid", label="python uncorrected spectrum of free axions")
plt.xlabel("k")
plt.ylabel("P(k)")
plt.xscale("log")
plt.yscale("log")
plt.title(f"log = {data.log_end:.2f}")
plt.legend()
plt.show()
