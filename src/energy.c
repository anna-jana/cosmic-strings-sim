#include <complex.h>
#include <stdlib.h>
#include <stdio.h>

#include "globals.h"


#define ENERGY_FNAME "energies.dat"
FILE* out_energies;
double* theta;
double* radial;

void init_energy_computation(void) {
    char* energy_filepath = create_output_filepath(ENERGY_FNAME);
    out_energies = fopen(energy_filepath, "w");
    theta = malloc(sizeof(double) * N3);
    radial = malloc(sizeof(double) * N3);
}

void deinit_energy_computation(void) {
    fclose(out_energies);
    free(theta);
    free(radial);
}

void compute_energy(void) {
    double a = TAU_TO_A(current_conformal_time);
    double a2 = a*a;

    for(int i = 0; i < N3; i++) {
        theta[i] = carg(phi[i]);
        radial[i] = sqrt(2) * cabs(phi[i]) / a - 1;
    }

    double mean_axion_kinetic = 0.0;
    double mean_axion_gradient = 0.0;
    double mean_radial_kinetic = 0.0;
    double mean_radial_gradient = 0.0;
    double mean_radial_potential = 0.0;
    double mean_interaction = 0.0;

    #pragma omp parallel for collapse(3)
    for(int iz = 0; iz < N; iz++) {
        for(int iy = 0; iy < N; iy++) {
            for(int ix = 0; ix < N; ix++) {
                const double R = creal(phi[AT(ix, iy, iz)]);
                const double Im = cimag(phi[AT(ix, iy, iz)]);
                const double R_dot = creal(phi_dot[AT(ix, iy, iz)]);
                const double I_dot = cimag(phi_dot[AT(ix, iy, iz)]);
                const double R2 = R*R;
                const double I2 = Im*Im;

                // axion
                // kinetic
                const double d_theta_d_tau = (I_dot * R - Im * R_dot) / (R2 - I2);
                const double axion_kinetic = 0.5 / a2 * d_theta_d_tau * d_theta_d_tau;
                // gradient
                const double diff_theta_x = theta[CYCLIC_AT(ix + 1, iy, iz)] - theta[CYCLIC_AT(ix - 1, iy, iz)];
                const double diff_theta_y = theta[CYCLIC_AT(ix, iy + 1, iz)] - theta[CYCLIC_AT(ix, iy - 1, iz)];
                const double diff_theta_z = theta[CYCLIC_AT(ix, iy, iz - 1)] - theta[CYCLIC_AT(ix, iy, iz - 1)];
                const double axion_gradient = 0.5 / (dx2) * (
                        diff_theta_x*diff_theta_x +
                        diff_theta_y*diff_theta_y +
                        diff_theta_z*diff_theta_z
                );

                // radial mode
                // kinetic
                const double d_r_d_tau = (R*R_dot + Im*I_dot) / cabs(phi[AT(ix, iy, iz)]);
                const double radial_kinetic = 0.5 / a2 * d_r_d_tau * d_r_d_tau;
                // gradient
                const double diff_radial_x = radial[CYCLIC_AT(ix + 1, iy, iz)] - radial[CYCLIC_AT(ix - 1, iy, iz)];
                const double diff_radial_y = radial[CYCLIC_AT(ix, iy + 1, iz)] - radial[CYCLIC_AT(ix, iy - 1, iz)];
                const double diff_radial_z = radial[CYCLIC_AT(ix, iy, iz - 1)] - radial[CYCLIC_AT(ix, iy, iz - 1)];
                const double radial_gradient = 0.5 / (dx2) * (
                        diff_radial_x*diff_radial_x +
                        diff_radial_y*diff_radial_y +
                        diff_radial_z*diff_radial_z
                );
                // potential
                const double r = radial[AT(ix, iy, iz)];
                const double r2 = r*r;
                const double inner = r2 - 2.0*r;
                const double radial_potential = inner * inner / 8.0;

                // interaction
                const double interaction = inner * (axion_kinetic + axion_gradient);

                mean_axion_kinetic += axion_kinetic;
                mean_axion_gradient += axion_gradient;
                mean_radial_kinetic += radial_kinetic;
                mean_radial_gradient += radial_gradient;
                mean_radial_potential += radial_potential;
                mean_interaction += interaction;
            }
        }
    }

    mean_axion_kinetic /= N3;
    mean_axion_gradient /= N3;
    mean_radial_kinetic /= N3;
    mean_radial_gradient /= N3;
    mean_radial_potential /= N3;
    mean_interaction /= N3;

    const double mean_axion_total = mean_axion_kinetic + mean_axion_gradient;
    const double mean_radial_total = mean_radial_kinetic + mean_radial_gradient + mean_radial_potential;
    const double mean_total = mean_axion_total + mean_radial_total + mean_interaction;

    fprintf(out_energies, "%i %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
            step,
            mean_axion_kinetic, mean_axion_gradient, mean_axion_total,
            mean_radial_kinetic, mean_radial_gradient, mean_radial_potential, mean_radial_total,
            mean_interaction, mean_total);
}
