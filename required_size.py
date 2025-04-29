import numpy as np
import matplotlib.pyplot as plt

log_ends = np.linspace(1.0, 9.0, 100)
def log_to_H(l): return 1.0 / np.exp(l)
L = 1 / log_to_H(log_ends)
required_N = np.ceil(L)
required_bytes = 4 * 4 * required_N**3 / (1024**3)

plt.plot(log_ends, required_bytes, label="required by simulation")
plt.axhline(8, color="k", ls="--", label="laptop")
plt.xlabel("scale log(m_r / H)")
plt.ylabel("giga bytes")
plt.yscale("log")
plt.legend()
plt.savefig("required_size.pdf")
plt.show()
