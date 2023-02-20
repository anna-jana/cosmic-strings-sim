#include "globals.h"

#include <complex.h>
#include <stdlib.h>

#define NBINS 20 // TODO: maybe make this a parameter later on
#define RADIUS 3
#define RADIUS2 (RADIUS*RADIUS)

static complex double* W;
static complex double* W_fft;
static complex double* theta_dot;
static complex double* theta_dot_fft;
static complex double* M;
static complex double* M_inv;
static double* spectrum;
static double* surface_integral_element;

static fftw_plan theta_dot_fft_plan;
static fftw_plan W_fft_plan;

void init_compute_spectrum(void) {
    W = malloc(sizeof(complex double) * N3);
    W_fft = malloc(sizeof(complex double) * N3);
    theta_dot = malloc(sizeof(complex double) * N3);
    theta_dot_fft = malloc(sizeof(complex double) * N3);
    M = malloc(sizeof(complex double) * NBINS * NBINS);
    M_inv = malloc(sizeof(complex double) * NBINS * NBINS);
    spectrum = malloc(sizeof(double) * NBINS);
    surface_integral_element = malloc(sizeof(double) * N);

    theta_dot_fft_plan = fftw_plan_dft_3d(N, N, N, theta_dot, theta_dot_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    W_fft_plan = fftw_plan_dft_3d(N, N, N, W, W_fft, FFTW_FORWARD, FFTW_ESTIMATE);
}

void deinit_compute_spectrum(void) {
    free(W);
    free(W_fft);
    free(theta_dot);
    free(theta_dot_fft);
    free(M);
    free(M_inv);
    free(spectrum);
    free(surface_integral_element);

    fftw_destroy_plan(theta_dot_fft_plan);
    fftw_destroy_plan(W_fft_plan);
}

static double compute_theta_dot(double a, complex double phi, complex double phi_dot) {
    const double R = creal(phi);
    const double Im = cimag(phi);
    const double R_dot = creal(phi_dot);
    const double I_dot = cimag(phi_dot);
    const double R2 = R*R;
    const double I2 = Im*Im;
    const double d_theta_d_tau = (I_dot * R - Im * R_dot) / (R2 - I2);
    return d_theta_d_tau / a;
}

static inline int idx_to_k(int idx) {
    return idx < N / 2 ? idx : -N/2 + idx  - N/2;
}

static inline int substract_wave_numbers(int i, int j, int N) {
    const int k = idx_to_k(i);
    const int k_prime = idx_to_k(j);
    const int k_diff = MOD(k - k_prime + N/2, N) - N/2;
    const int idx_diff = k_diff < 0 ? k_diff + N : k_diff;
    return idx_diff;
}

// needs to be called after detect_strings()
void compute_spectrum(void) {
    // compute scale factor
    const double a = TAU_TO_A(current_conformal_time);

    // compute the theta dot field
    for(int i = 0; i < N3; i++) {
        theta_dot[i] = compute_theta_dot(a, phi[i], phi_dot[i]);
    }

    // compute W
    for(int i = 0; i < N3; i++) {
        W[i] = 1 + 1*I;
    }
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {
        for(int i = 0; i < points_lengths[thread_id]; i++) {
            // set sphere of RADIUS RADIUS around p to 1
            for(int x_offset = -RADIUS; x_offset <= RADIUS; x_offset++) {
                for(int y_offset = -RADIUS; y_offset <= RADIUS; y_offset++) {
                    for(int z_offset = -RADIUS; z_offset <= RADIUS; z_offset++) {
                        const double r2 = x_offset*x_offset + y_offset*y_offset + z_offset*z_offset;
                        const struct Index p = points[thread_id][i];
                        if(r2 < RADIUS2) {
                            W[CYCLIC_AT(p.ix + x_offset, p.iy + y_offset, p.iz + z_offset)] = 0.0 + 0.0*I;
                        }
                    }
                }
            }
        }
    }

    // mask the strings out by multiplying point wise with W
    for(int i = 0; i < N3; i++) {
        theta_dot[i] *= W[i];
    }

    // calc histogram quantaties and integration measures
    const double dx_physical = dx * a;
    const double kmax = calc_k_max_grid(N, dx_physical);
    const double Delta_k = 2*PI/dx_physical;
    const double bin_width = kmax / NBINS;
    for(int i = 0; i < NBINS; i++) {
        const double vol = 4.0/3.0 * PI * (pow((i + 1)*bin_width, 3) - pow(i*bin_width, 3));
        const double area = 4*PI * pow(i*bin_width + bin_width/2, 2);
        surface_integral_element[i] = area / vol * pow(Delta_k, 3);
    }

    // fft of W*dot theta
    fftw_execute(theta_dot_fft_plan);

    // spectrum of W*dot theta
    for(int i = 0; i < NBINS; i++) {
        spectrum[i] = 0.0;
        const double bin_k_min = i * bin_width;
        const double bin_k_max = bin_k_min + bin_width;
        const double bin_k = bin_k_min + bin_width/2.0;
        for(int iz = 0; iz < N; iz++) {
            for(int iy = 0; iy < N; iy++) {
                for(int ix = 0; ix < N; ix++) {
                    const double kx = ks[ix];
                    const double ky = ks[iy];
                    const double kz = ks[iz];
                    const double k2 = kx*kx + ky*ky + kz*kz;
                    if(k2 >= bin_k_min*bin_k_min &&
                       k2 <= bin_k_max*bin_k_max) {
                        const double f = theta_dot_fft[AT(ix, iy, iz)];
                        spectrum[i] += creal(f)*creal(f) + cimag(f)*cimag(f);
                    }
                }
            }
        }
        spectrum[i] *= surface_integral_element[i];
        spectrum[i] *= bin_k*bin_k / (L*L*L) / (4 * PI) * 0.5;
    }

    // W fft
    fftw_execute(W_fft_plan);

    // compute M
    for(int i = 0; i < NBINS; i++) {
        for(int j = 0; j < NBINS; j++) {
        }
    }
    // invert M
    // multiply spectrum by M
    // output spectrum
}
