#include "globals.h"

#define ENERGY_FNAME "energies.dat"
FILE* out_energies;

void init_energy_computation(void) {
    out_energies = fopen(ENERGY_FNAME, "w");
}

void deinit_energy_computation(void) {
    fclose(out_energies);
}

void compute_energy(void) {
    double axion_kinetic = 0.0;
    double axion_gradient = 0.0;
    double radial_kinetic = 0.0;
    double radial_gradient = 0.0;
    double radial_potential = 0.0;
    double interaction = 0.0;

    for(int ix = 0; ix < N; ix++) {
        for(int iy = 0; iy < N; iy++) {
            for(int iz = 0; iz < N; iz++) {
            }
        }
    }

    double axion_total = axion_kinetic + axion_gradient;
    double radial_total = radial_kinetic + radial_gradient + radial_potential;
    double total = axion_total + radial_total + interaction;

    fprintf(out_energies, "%i %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
            step,
            axion_kinetic, axion_gradient, axion_total,
            radial_kinetic, radial_gradient, radial_potential, radial_total,
            interaction, total);
}
