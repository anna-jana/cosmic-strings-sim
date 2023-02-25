#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <complex.h>
// fftw3_complex is complex double bc
// complex.h is included before fftw3.h
#include <fftw3.h>

#include "globals.h"

void make_step(void) {
    current_conformal_time = TAU_START + (step + 1) * Delta_tau;

    // propagate PDE using velocity verlet algorithm
    // update the field ("position")
    #pragma omp parallel for
    for(int i = 0; i < N3; i++)
        next_phi[i] = phi[i] + Delta_tau*phi_dot[i] + 0.5*Delta_tau*Delta_tau*phi_dot_dot[i];
    // update the field derivative ("velocity")
    compute_next_force();
    #pragma omp parallel for
    for(int i = 0; i < N3; i++)
        next_phi_dot[i] = phi_dot[i] + Delta_tau*(phi_dot_dot[i] + next_phi_dot_dot[i])/2;

    // swap current and next arrays
    fftw_complex* tmp;
    // swap phi and next_phi
    tmp = phi;
    phi = next_phi;
    next_phi = tmp;
    // swap phi_dot and next_phi_dot
    tmp = phi_dot;
    phi_dot = next_phi_dot;
    next_phi_dot = tmp;
    // swap phi_dot_dot and next_phi_dot_dot
    tmp = phi_dot_dot;
    phi_dot_dot = next_phi_dot_dot;
    next_phi_dot_dot = tmp;
}

// calculate the right hand side of the PDE
void compute_next_force(void) {
    double scale_factor = TAU_TO_A(current_conformal_time);
    #pragma omp parallel for collapse(3)
    for(int iz = 0; iz < N; iz++) {
        for(int iy = 0; iy < N; iy++) {
            for(int ix = 0; ix < N; ix++) {
                const complex double phi_val = phi[AT(ix, iy, iz)];
                const double phi_abs = creal(phi_val) * creal(phi_val) + cimag(phi_val) * cimag(phi_val);
                const complex double pot_force = phi_val * (phi_abs - 0.5*scale_factor);
                const complex double laplace = - 6 * phi_val +
                    phi[CYCLIC_AT(ix + 1, iy, iz)] +
                    phi[CYCLIC_AT(ix - 1, iy, iz)] +
                    phi[CYCLIC_AT(ix, iy + 1, iz)] +
                    phi[CYCLIC_AT(ix, iy - 1, iz)] +
                    phi[CYCLIC_AT(ix, iy, iz + 1)] +
                    phi[CYCLIC_AT(ix, iy, iz - 1)];
                next_phi_dot_dot[AT(ix, iy, iz)] = + laplace - pot_force;
            }
        }
    }
}

