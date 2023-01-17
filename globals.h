#ifndef GLOBALS_H
#define GLOBALS_H

#include <math.h>
#include <complex.h>

#include <fftw3.h>

#define PI 3.14159265358979323846

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

/*************************************** cosmology functions ********************************/
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

// utils.c
double* fft_freq(int n, double d);
double calc_k_max_grid(int n, double d);
double random_uniform(double min, double max);
int mod(int a, int b);

// propagator.c
// simulation state (global for now, maybe put into struct later)
extern double current_conformal_time;
extern int i;
extern fftw_complex *phi, *phi_dot, *phi_dot_dot;
extern fftw_complex *next_phi, *next_phi_dot, *next_phi_dot_dot;

// initial state generation
extern fftw_complex *hat;
extern double* ks;

void init(void);
void deinit(void);
void step(void);
void random_field(fftw_complex* field);
void compute_next_force(void);

#endif // GLOBALS_H
