#ifndef GLOBALS_H
#define GLOBALS_H

#include <math.h>
#include <complex.h>

#include <fftw3.h>

#define DEBUG
#define EVERY_ANALYSIS_STEP 10

// parameters
// simulation domain in time in log units
#define LOG_START 2.0
#define LOG_END 3.0
#define L (1/LOG_TO_H(LOG_END)) // comoving length of the simulation box in units of 1/m_r
#define N ((int)ceil(L * TAU_TO_A(LOG_TO_TAU(LOG_END)))) // number of grid points in one dimension
#define DELTA -1e-2 // step size for time stepping

/******************************** utils.c **************************/
#define PI 3.14159265358979323846
double* fft_freq(int n, double d);
double calc_k_max_grid(int n, double d);
double random_uniform(double min, double max);
int mod(int a, int b);
int sign(double x);

/*** cosmology functions ***/
// functions for converting between cosmological variables
// all variables have units in powers of m_r
#define LOG_TO_H(LOG) (1/exp(LOG))
#define H_TO_T(H) (1 / (2*(H)))
#define T_TO_H(T) (1 / (2*(T)))
#define H_TO_LOG(H) (log(1/(H)))
#define T_TO_TAU(T) (-2*sqrt(T))
#define LOG_TO_TAU(LOG) (T_TO_TAU(H_TO_T(LOG_TO_H(LOG))))
#define T_TO_A(T) (sqrt(T))
#define TAU_TO_T(TAU) pow(-0.5*(TAU), 2)
#define TAU_TO_A(TAU) (-0.5*TAU)
#define TAU_TO_LOG(TAU) H_TO_LOG(T_TO_H(TAU_TO_T(TAU)))

void write_field(char* fname);
#define PARAMETER_FILENAME "parameter.json"
void output_parameters(void);


/****************************** propagator.c *************************/
/*** spacial discretisation ***/
// for N see parameters, number of grid points in one dimension
// total number
#define N3 (N*N*N)
#define AT(ix, iy, iz) ((ix) + (iy) * (N) + (iz) * (N) * (N))
#define CYCLIC_AT(ix, iy, iz) AT(mod(ix, N), mod(iy, N), mod(iz, N))
// for L see parameters, comoving length of the simulation box in units of 1/m_r
// L/N not L/(N-1) bc we have cyclic boundary conditions
// *...*...* N = 2, dx = L / N
#define dx (L/N)
#define dx2 (dx*dx)
extern fftw_complex *phi, *phi_dot, *phi_dot_dot;
extern fftw_complex *next_phi, *next_phi_dot, *next_phi_dot_dot;

/*** simulation time ***/
// simulation domain in time in conformal time
// for LOG_START and LOG_END see parameters
#define TAU_START LOG_TO_TAU(LOG_START)
#define TAU_END LOG_TO_TAU(LOG_END)
#define TAU_SPAN ((TAU_END) - (TAU_START))
// discretisation (DELTA is negative because the conformal time is decreasing
// for DELTA see parameters
#define NSTEPS ((int)(ceil(TAU_SPAN / DELTA)))
extern double current_conformal_time;
extern int step;

/*** initial state generation ***/
#define FIELD_MAX (1 / sqrt(2))
#define KMAX 1.0
#define SEED 42
extern fftw_complex *hat;
extern double* ks;
void random_field(fftw_complex* field);

/*** propagation ***/
void init(void);
void deinit(void);
void make_step(void);
void compute_next_force(void);

/********************************* string_detection.c ****************************/
/*** detect and store strings ***/
void init_detect_strings(void);
void deinit_detect_strings(void);
void detect_strings(void);

/********************************* energy.c *********************************/
void init_energy_computation(void);
void deinit_energy_computation(void);
void compute_energy(void);

#endif // GLOBALS_H
