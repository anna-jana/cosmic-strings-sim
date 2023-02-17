#ifndef GLOBALS_H
#define GLOBALS_H

#include <math.h>
#include <complex.h>
#include <stdbool.h>

#include <fftw3.h>

#define DEBUG

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

/********************************* io.c ********************************/
void write_field(char* fname, const complex double* field);
void output_parameters(void);
void create_output_dir(void);
char* create_output_filepath(const char* filename);
bool parse_arg(int argc, char* argv[], char* arg, char type, bool required, void* value);

/********************************** state.c ******************************/
// parameters
extern double LOG_START, LOG_END;
extern double L;
extern int N;
extern double Delta_tau;
extern double KMAX;
extern int SEED;
extern int EVERY_ANALYSIS_STEP;

extern int N3;
extern double dx, dx2;
extern double TAU_START;
extern double TAU_END;
extern double TAU_SPAN;
extern int NSTEPS;

/*** spacial discretisation ***/
extern double current_conformal_time; // simulation domain in time in conformal time
extern int step;
extern fftw_complex *phi, *phi_dot, *phi_dot_dot;
extern fftw_complex *next_phi, *next_phi_dot, *next_phi_dot_dot;

// for N see parameters, number of grid points in one dimension
#define AT(ix, iy, iz) ((ix) + (iy) * (N) + (iz) * (N) * (N))
#define CYCLIC_AT(ix, iy, iz) AT(mod(ix, N), mod(iy, N), mod(iz, N))

void init_parameters(int argc, char* argv[]);
void init_state(void);
void deinit_state(void);

/****************************** propagator.c *************************/
void make_step(void);
void compute_next_force(void);

/********************************* string_detection.c ****************************/
void init_detect_strings(void);
void deinit_detect_strings(void);
void detect_strings(void);

/************************************* energy.c *********************************/
void init_energy_computation(void);
void deinit_energy_computation(void);
void compute_energy(void);

/************************************* spectrum.c *******************************/
void init_compute_spectrum(void);
void deinit_compute_spectrum(void);
void compute_spectrum(void);

#endif // GLOBALS_H
