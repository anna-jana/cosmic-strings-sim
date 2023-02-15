import json, sys, itertools, os
import numpy as np
import cosmology

class OutputDir:
    def __init__(self, dirname):
        self.dirname = dirname

        with open(self.create_output_path("parameter.json"), "r") as f:
            parameter = json.load(f)

        self.log_start = parameter["LOG_START"]
        self.log_end = parameter["LOG_END"]
        self.L = parameter["L"]
        self.N = parameter["N"]
        self.dtau = parameter["DELTA"]

        self.tau_start = cosmology.log_to_tau(self.log_start)
        self.tau_end = cosmology.log_to_tau(self.log_end)
        self.dx = self.L / self.N

        self.final_field = self.load_field("final_field.dat")
        self.final_field_dot = self.load_field("final_field_dot.dat")

        self.string_step, self.string_id, self.string_x, self.string_y, self.string_z = \
                np.loadtxt(self.create_output_path("strings.dat")).T

        self.energy_step, self.axion_kinetic, self.axion_gradient, self.axion_total, self.radial_kinetic, \
            self.radial_gradient, self.radial_potential, self.radial_total, self.interaction, self.total = \
            np.loadtxt(self.create_output_path("energies.dat")).T

    def create_output_path(self, fname):
        return os.path.join(self.dirname, fname)

    def load_field(self, fname):
        field = np.loadtxt(self.create_output_path(fname), dtype="complex")
        return field.ravel().reshape(self.N, self.N, self.N).transpose(2,1,0)



