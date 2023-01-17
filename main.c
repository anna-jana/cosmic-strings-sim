#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>

#include <fftw3.h>
// fftw3_complex is complex double bc
// complex.h is included before fftw3.h

#define PI 3.14159265358979323846

static inline int mod(int a, int b) {
    return ((a % b) + b) % b;
}

// spacial discretisation
// number of grid points in one dimension
#define N 20
// total number
#define N3 (N*N*N)
#define AT(ix, iy, iz) ((ix) + (iy) * (N) + (iz) * (N) * (N))
#define CYCLIC_AT(ix, iy, iz) AT(mod(ix, N), mod(iy, N), mod(iz, N))
// comoving length of the simulation box in units of 1/m_r
#define L 1.0
// L/N not L/(N-1) bc we have cyclic boundary conditions
// *...*...* N = 2, dx = L / N
#define dx (L/N)

// simulation time
// functions for converting between cosmological variables
// all variables have units in powers of m_r
#define LOG_TO_H(LOG) (1/exp(LOG))
#define H_TO_T(H) (1 / (2*(H)))
#define T_TO_H(T) (1 / (2*(T)))
#define H_TO_LOG(H) (log(1/H))
#define T_TO_TAU(T) (-2*sqrt(T))
#define LOG_TO_TAU(LOG) (T_TO_TAU(H_TO_T(LOG_TO_H(LOG))))
#define T_TO_A(T) (sqrt(T))
#define TAU_TO_T(TAU) pow(-0.5*(TAU), 2)
#define TAU_TO_A(TAU) (-0.5*TAU)
#define TAU_TO_LOG(TAU) H_TO_LOG(T_TO_H(TAU_TO_T(TAU)))
// simulation domain in time in log units
#define LOG_START 2
#define LOG_END 4
// simulation domain in time in conformal time
#define TAU_START LOG_TO_TAU(LOG_START)
#define TAU_END LOG_TO_TAU(LOG_END)
#define TAU_SPAN ((TAU_END) - (TAU_START))
// discretisation (DELTA is negative because the conformal time is decreasing
#define DELTA -1e-2
#define NSTEPS ((int)(ceil(TAU_SPAN / DELTA)))

// initial field generation
#define FIELD_MAX (1 / sqrt(2))
#define KMAX 0.3

// generate an array with the frequencies (in our case wavenumber) returned by the discrete fourier transform
double* fft_freq(int n, double d) {
    double* freq = malloc(sizeof(double) * n);
    if(n % 2 == 0) {
        for(int i = 0; i <= n / 2 - 1; i++)
            freq[i] = i / (d*n);
        for(int i = n / 2; i < n; i++)
            freq[i] = (- n / 2 + (i - n/2)) / (d*n);
    } else {
        for(int i = 0; i <= (n - 1) / 2; i++)
            freq[i] = i / (d*n);
        for(int i = (n - 1) / 2 + 1; i < n; i++)
            freq[i] = (-(n - 1) / 2 + (i - (n - 1) / 2)) / (d*n);
    }
    for(int i = 0; i < n; i++)
        freq[i] *= 2 * PI;
    return freq;
}

// calculate the maximal frequency (in our case wavenumber) on the grid
double calc_k_max_grid(int n, double d) {
    if(n % 2 == 0) {
        return 2 * PI * (n / 2 - 1) / (d*n);
    } else {
        return 2 * PI * ((n - 1) / 2) / (d*n);
    }
}

// random double between min and max
double random_uniform(double min, double max) {
    return min + rand() / (double) RAND_MAX * (max - min);
}

// simulation state (global for now, maybe put into struct later)
double current_conformal_time;
int i;
fftw_complex *phi, *phi_dot, *phi_dot_dot;
fftw_complex *next_phi, *next_phi_dot, *next_phi_dot_dot;

// initial state generation
fftw_complex *hat;
double* ks;

// generate random complex field with strings
void random_field(fftw_complex* field) {
    double kmax_grid = calc_k_max_grid(N, dx);
    for(int ix = 0; ix < N; ix++) {
        for(int iy = 0; iy < N; iy++) {
            for(int iz = 0; iz < N; iz++) {
                double kx = ks[ix];
                double ky = ks[iy];
                double kz = ks[iz];
                double k = sqrt(kx*kx + ky*ky + kz*kz);
                if(k <= KMAX*kmax_grid) {
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

void init(void) {
    printf("running simuation\n");
    assert(KMAX >= 0.0 && KMAX <= 1.0);

    current_conformal_time = TAU_START;
    i = 0;

    phi = fftw_malloc(sizeof(fftw_complex) * N3);
    phi_dot = fftw_malloc(sizeof(fftw_complex) * N3);
    phi_dot_dot = fftw_malloc(sizeof(fftw_complex) * N3);
    next_phi = fftw_malloc(sizeof(fftw_complex) * N3);
    next_phi_dot = fftw_malloc(sizeof(fftw_complex) * N3);
    next_phi_dot_dot = fftw_malloc(sizeof(fftw_complex) * N3);
    hat = fftw_malloc(sizeof(fftw_complex) * N3);

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
    printf("\ndone\n");
}

void step(void) {
    printf("\rstep: %i, conformal time: %lf, log: %lf", i, current_conformal_time, TAU_TO_LOG(current_conformal_time));
    current_conformal_time = TAU_START + i * DELTA;

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

void write_slice(char* fname) {
    printf("\nwriting slice\n");
    FILE* out = fopen(fname, "w");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            fprintf(out, "%lf+%lfj ", creal(phi[AT(i, j, 0)]), cimag(phi[AT(i, j, 0)]));
        }
        fprintf(out, "\n");
    }
    fclose(out);
}

int main(int argc, char* argv[]) {
    init();
    write_slice("initial_slice.dat");
    for(i = 0; i < NSTEPS; i++)
        step();
    write_slice("final_slice.dat");
    deinit();
    return EXIT_SUCCESS;
}

