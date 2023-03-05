#include "globals.h"

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <fftw3.h>
#include <omp.h>

// parameters
double LOG_START, LOG_END; // simulation domain in time in log units
double L; // comoving length of the simulation box in units of 1/m_r
int N; // number of grid points in one dimension
double Delta_tau;
int SEED;
double KMAX;
int EVERY_ANALYSIS_STEP;

// derived parameters
int N3;
double dx, dx2;
double TAU_START;
double TAU_END;
double TAU_SPAN;
int NSTEPS;

// runtime parameters
int num_threads;

// simulation state (global for now, maybe put into struct later)
double current_conformal_time;
int step;
complex double *phi, *phi_dot, *phi_dot_dot;
complex double *next_phi, *next_phi_dot, *next_phi_dot_dot;

// initial state generation
#define FIELD_MAX (1 / sqrt(2))
static fftw_complex *hat;
double* ks;

void init_parameters(int argc, char* argv[]) {
    // determine number of threads to use
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            printf("INFO: using %i threads\n", num_threads);
        }
    }

    // setting default values
    LOG_START = 2.0;
    LOG_END = 3.0;
    Delta_tau = -1e-2; // step size for time stepping
    KMAX = 1.0;
    SEED = 42;
    EVERY_ANALYSIS_STEP = 10;

    parse_arg(argc, argv, "--Delta-tau", 'f', false, &Delta_tau);
    parse_arg(argc, argv, "--log-start", 'f', false, &LOG_START);
    parse_arg(argc, argv, "--log-end",   'f', false, &LOG_END);
    parse_arg(argc, argv, "--kmax",      'f', false, &KMAX);
    parse_arg(argc, argv, "--seed",      'i', false, &SEED);
    parse_arg(argc, argv, "--every-analysis-step", 'i', false, &EVERY_ANALYSIS_STEP);

    // calculate N and L from other parameters if not set explicitly
    bool L_given = parse_arg(argc, argv, "--L", 'f', false, &L);
    if(!L_given) {
        L = 1 / LOG_TO_H(LOG_END);
    }

    bool N_given = parse_arg(argc, argv, "--N", 'i', false, &N);
    if(!N_given) {
        N = (int) ceil(L * TAU_TO_A(LOG_TO_TAU(LOG_END)));
    }

    // setting derived parameters
    N3 = N*N*N; // total number of grid points
    // L/N not L/(N-1) bc we have cyclic boundary conditions*...*...* N = 2
    dx = L / N;
    dx2 = dx*dx;
    TAU_START = LOG_TO_TAU(LOG_START);
    TAU_END = LOG_TO_TAU(LOG_END);
    TAU_SPAN = TAU_END - TAU_START;
    NSTEPS = (int) ceil(TAU_SPAN / Delta_tau);
}

// generate random complex field with strings
static void random_field(fftw_complex* field) {
    for(int iz = 0; iz < N; iz++) {
        for(int iy = 0; iy < N; iy++) {
            for(int ix = 0; ix < N; ix++) {
                const double kx = ks[ix];
                const double ky = ks[iy];
                const double kz = ks[iz];
                const double k = sqrt(kx*kx + ky*ky + kz*kz);
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
    #pragma omp parallel for
    for(int i = 0; i < N3; i++)
        field[i] /= N;
}

void init_state(void) {
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
    printf("INFO: kmax_grid = %lf, KMAX = %lf\n", kmax_grid, KMAX);
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

void deinit_state(void) {
    fftw_free(phi);
    fftw_free(phi_dot);
    fftw_free(phi_dot_dot);
    fftw_free(next_phi);
    fftw_free(next_phi_dot);
    fftw_free(next_phi_dot_dot);
    fftw_free(hat);
    free(ks);
}

