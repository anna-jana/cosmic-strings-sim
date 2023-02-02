#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <complex.h>
// fftw3_complex is complex double bc
// complex.h is included before fftw3.h
#include <fftw3.h>

#include "globals.h"
// simulation state (global for now, maybe put into struct later)
double current_conformal_time;
int step;
fftw_complex *phi, *phi_dot, *phi_dot_dot;
fftw_complex *next_phi, *next_phi_dot, *next_phi_dot_dot;

// initial state generation
fftw_complex *hat;
double* ks;

void init(void) {
    srand(SEED);

    current_conformal_time = TAU_START;
    step = 0;

    phi = fftw_malloc(sizeof(fftw_complex) * N3);
    phi_dot = fftw_malloc(sizeof(fftw_complex) * N3);
    phi_dot_dot = fftw_malloc(sizeof(fftw_complex) * N3);
    next_phi = fftw_malloc(sizeof(fftw_complex) * N3);
    next_phi_dot = fftw_malloc(sizeof(fftw_complex) * N3);
    next_phi_dot_dot = fftw_malloc(sizeof(fftw_complex) * N3);
    hat = fftw_malloc(sizeof(fftw_complex) * N3);

    double kmax_grid = calc_k_max_grid(N, dx);
    printf("kmax_grid: %lf, KMAX: %lf\n", kmax_grid, KMAX);
    fflush(stdout);
    assert(KMAX >= 0.0 && KMAX <= kmax_grid);
    ks = fft_freq(N, dx);
    random_field(phi);
    random_field(phi_dot);

    compute_next_force();
    // if have to swap phi_dot_dot and next_phi_dot_dot,
    // bc compute_next_force() is wrting to the next_phi_dot_dot array
    fftw_complex* tmp = phi_dot_dot;
    phi_dot_dot = next_phi_dot_dot;
    next_phi_dot_dot = tmp;
}

void deinit(void) {
    fftw_free(phi);
    fftw_free(phi_dot);
    fftw_free(phi_dot_dot);
    fftw_free(next_phi);
    fftw_free(next_phi_dot);
    fftw_free(next_phi_dot_dot);
    fftw_free(hat);
    free(ks);
}

void make_step(void) {
    current_conformal_time = TAU_START + step * DELTA;

    // propagate PDE using velocity verlet algorithm
    // update the field ("position")
    for(int i = 0; i < N3; i++)
        next_phi[i] = phi[i] + DELTA*phi_dot[i] + 0.5*DELTA*DELTA*phi_dot_dot[i];
    // update the field derivative ("velocity")
    compute_next_force();
    for(int i = 0; i < N3; i++)
        next_phi_dot[i] = phi_dot[i] + DELTA*(phi_dot_dot[i] + next_phi_dot_dot[i])/2;

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

// generate random complex field with strings
void random_field(fftw_complex* field) {
    for(int ix = 0; ix < N; ix++) {
        for(int iy = 0; iy < N; iy++) {
            for(int iz = 0; iz < N; iz++) {
                double kx = ks[ix];
                double ky = ks[iy];
                double kz = ks[iz];
                double k = sqrt(kx*kx + ky*ky + kz*kz);
                if(k <= KMAX) {
                    hat[AT(ix, iy, iz)] =
                        random_uniform(- FIELD_MAX, FIELD_MAX);
                } else {
                    hat[AT(ix, iy, iz)] = 0.0;
                }
            }
        }
    }
    fftw_plan gen_plan = fftw_plan_dft_3d(N, N, N, hat, field, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(gen_plan);
    fftw_destroy_plan(gen_plan);
    for(int i = 0; i < N3; i++)
        field[i] /= N;
}

// calculate the right hand side of the PDE
void compute_next_force(void) {
    double scale_factor = TAU_TO_A(current_conformal_time);
    for(int ix = 0; ix < N; ix++) {
        for(int iy = 0; iy < N; iy++) {
            for(int iz = 0; iz < N; iz++) {
                complex double phi_val = phi[AT(ix, iy, iz)];
                double phi_abs = creal(phi_val) * creal(phi_val) + cimag(phi_val) * cimag(phi_val);
                complex double pot_force = phi_val * (phi_abs - 0.5*scale_factor);
                complex double laplace = - 6 * phi_val +
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

