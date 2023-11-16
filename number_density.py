import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def f_gorghetto(l, q):
    return (1 - 1/q) / (1 - (2*q - 1) * np.exp((1 - q) * l))


def integrant(l, q):
    return l * np.exp(l/2) / (1 - np.exp((1 - q)*l)) * (1 - np.exp(-q*l))


def calc_f_numerical(l, q):
    I = quad(integrant, -10, l, args=(q,))[0]
    return 0.5 * (1 - 1 / q) * np.exp(-0.5*l) / l * I


qs = np.linspace(0.5, 5.0, 200)[1:]
l = 70
plt.plot(qs, f_gorghetto(l, qs), label="Gorghettos Formula")
plt.plot(qs, np.array([calc_f_numerical(l, q)
         for q in qs]), label="numerical integral")
plt.yscale("log")
plt.ylim(1e-4, 1)
plt.xlabel("q")
plt.ylabel(r"$f(q) = n_a / (8 H \xi \mu / x_0)$")
plt.legend()
plt.title(f"log = {l}")
plt.grid()
