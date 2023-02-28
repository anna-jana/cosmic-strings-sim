#include "globals.h"

#include <complex.h>
#include <stdlib.h>
#include <stdio.h>

#include <fftw3.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>

#define NBINS 20 // TODO: maybe make this a parameter later on
#define RADIUS 2 // TODO: tune this parameter
#define RADIUS2 (RADIUS*RADIUS)

static complex double* W;
static complex double* W_fft;
static complex double* theta_dot;
static complex double* theta_dot_fft;

static double* physical_ks;

static struct Index** spheres;
static int* sphere_list_capacities;
static int* sphere_list_lengths;

static double* surface_integral_element;
static double* spectrum_uncorrected;
static double* spectrum_corrected;

static fftw_plan theta_dot_fft_plan; static fftw_plan W_fft_plan;
static gsl_matrix* M;
static gsl_matrix* M_inv;
static gsl_permutation* p;

static FILE* out;

void init_compute_spectrum(void) {
    W = malloc(sizeof(complex double) * N3);
    W_fft = malloc(sizeof(complex double) * N3);
    theta_dot = malloc(sizeof(complex double) * N3);
    theta_dot_fft = malloc(sizeof(complex double) * N3);

    physical_ks = malloc(sizeof(double) * N);

    spheres = malloc(sizeof(struct Index*) * NBINS);
    sphere_list_capacities = malloc(sizeof(int) * NBINS);
    sphere_list_lengths = malloc(sizeof(int) * NBINS);
    for(int i = 0; i < NBINS; i++) {
        sphere_list_capacities[i] = N;
        spheres[i] = malloc(sizeof(struct Index) * sphere_list_capacities[i]);
    }

    surface_integral_element = malloc(sizeof(double) * NBINS);
    spectrum_uncorrected = malloc(sizeof(double) * NBINS);
    spectrum_corrected = malloc(sizeof(double) * NBINS);

    M = gsl_matrix_alloc(NBINS, NBINS);
    M_inv = gsl_matrix_alloc(NBINS, NBINS);
    p = gsl_permutation_alloc(NBINS);

    theta_dot_fft_plan = fftw_plan_dft_3d(N, N, N, theta_dot, theta_dot_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    W_fft_plan = fftw_plan_dft_3d(N, N, N, W, W_fft, FFTW_FORWARD, FFTW_ESTIMATE);

    const char* fname = create_output_filepath("spectrum.dat");
    printf("INFO: writing spectrum to %s\n", fname);
    out = fopen(fname, "w");
}

void deinit_compute_spectrum(void) {
    free(W);
    free(W_fft);
    free(theta_dot);
    free(theta_dot_fft);

    free(physical_ks);

    for(int i = 0; i < NBINS; i++) {
        free(spheres[i]);
    }
    free(spheres);
    free(sphere_list_capacities);
    free(sphere_list_lengths);

    free(spectrum_uncorrected);
    free(spectrum_corrected);
    free(surface_integral_element);

    gsl_matrix_free(M);
    gsl_matrix_free(M_inv);
    gsl_permutation_free(p);

    fftw_destroy_plan(theta_dot_fft_plan);
    fftw_destroy_plan(W_fft_plan);

    fclose(out);
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

static inline int substract_wave_numbers(int i, int j) {
    const int k = idx_to_k(i);
    const int k_prime = idx_to_k(j);
    const int k_diff = mod(k - k_prime + N/2, N) - N/2;
    const int idx_diff = k_diff < 0 ? k_diff + N : k_diff;
    return idx_diff;
}

// needs to be called after detect_strings()
void compute_spectrum(void) {
    // compute scale factor
    const double a = TAU_TO_A(current_conformal_time);

    // compute the theta dot field
    #pragma omp parallel for
    for(int i = 0; i < N3; i++) {
        theta_dot[i] = compute_theta_dot(a, phi[i], phi_dot[i]);
    }
#ifdef DEBUG
    printf("DEBUG: output of theta_dot\n");
    FILE* theta_dot_out = fopen(create_output_filepath("theta_dot.dat"), "w");
    for(int i = 0; i < N3; i++) {
        fprintf(theta_dot_out, "%.15e\n", creal(theta_dot[i]));
    }
    fclose(theta_dot_out);
#endif

    // compute W
    #pragma omp parallel for
    for(int i = 0; i < N3; i++) {
        W[i] = 1 + 1*I;
    }
    #pragma omp parallel for
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {
        for(int i = 0; i < last_points_lengths[thread_id]; i++) {
            // set sphere of RADIUS RADIUS around p to 1
            for(int x_offset = -RADIUS; x_offset <= RADIUS; x_offset++) {
                for(int y_offset = -RADIUS; y_offset <= RADIUS; y_offset++) {
                    for(int z_offset = -RADIUS; z_offset <= RADIUS; z_offset++) {
                        const double r2 = x_offset*x_offset + y_offset*y_offset + z_offset*z_offset;
                        const struct Index p = points[thread_id][i];
                        if(r2 <= RADIUS2) {
                            W[CYCLIC_AT(p.ix + x_offset, p.iy + y_offset, p.iz + z_offset)] = 0.0 + 0.0*I;
                        }
                    }
                }
            }
        }
    }
#ifdef DEBUG
    printf("DEBUG: output of W\n");
    printf("DEBUG: RADIUS2 = %i, RADIUS = %i\n", RADIUS2, RADIUS);
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {
        printf("DEBUG: last_points_lengths[%i] = %i\n", thread_id, last_points_lengths[thread_id]);
    }
    fflush(stdout);
    FILE* W_out = fopen(create_output_filepath("W.dat"), "w");
    for(int i = 0; i < N3; i++) {
        fprintf(W_out, "%.15e\n", creal(W[i]));
    }
    fclose(W_out);
#endif

    // mask the strings out by multiplying point wise with W
    #pragma omp parallel for
    for(int i = 0; i < N3; i++) {
        theta_dot[i] *= W[i];
    }

    // calc histogram quantaties and integration measures
    const double dx_physical = dx * a;
    const double kmax = calc_k_max_grid(N, dx_physical);
    const double Delta_k = 2*PI/dx_physical;
    const double bin_width = kmax / NBINS;

#ifdef DEBUG
    printf("DEBUG: current_conformal_time = %lf\n", current_conformal_time);
    printf("DEBUG: dx = %lf, a = %lf, NBINS = %i\n", dx, a, NBINS);
    printf("DEBUG: dx_physical = %lf, kmax = %lf, Delta_k = %lf, bin_width = %lf\n",
            dx_physical, kmax, Delta_k, bin_width);
#endif

    #pragma omp parallel for
    for(int i = 0; i < NBINS; i++) {
        const double vol = 4.0/3.0 * PI * (pow((i + 1)*bin_width, 3) - pow(i*bin_width, 3));
        const double area = 4*PI * pow(i*bin_width + bin_width/2, 2);
        surface_integral_element[i] = area / vol * pow(Delta_k, 3);
    }

    // fft of W*dot theta
    fftw_execute(theta_dot_fft_plan);

    // compute surface integration spheres
    // TODO: these should be global
    fill_fft_freq(N, dx_physical, physical_ks);
    #pragma omp parallel for
    for(int i = 0; i < NBINS; i++) {
        sphere_list_lengths[i] = 0;
        const double bin_k_min = i * bin_width;
        const double bin_k_max = bin_k_min + bin_width;
        for(int iz = 0; iz < N; iz++) {
            for(int iy = 0; iy < N; iy++) {
                for(int ix = 0; ix < N; ix++) {
                    const double kx = physical_ks[ix];
                    const double ky = physical_ks[iy];
                    const double kz = physical_ks[iz];
                    const double k2 = kx*kx + ky*ky + kz*kz;
                    if(k2 >= bin_k_min*bin_k_min &&
                       k2 <= bin_k_max*bin_k_max) {
                        if(sphere_list_lengths[i] >= sphere_list_capacities[i]) {
                            sphere_list_capacities[i] *= 2;
                            spheres[i] = realloc(spheres[i], sizeof(struct Index) * sphere_list_capacities[i]);
                        }
                        const struct Index index = {ix, iy, iz};
                        spheres[i][sphere_list_lengths[i]++] = index;
                    }
                }
            }
        }
    }
#ifdef DEBUG
    for(int i = 0; i < NBINS; i++) {
        printf("DEBUG: #points of sphere[%i] = %i\n", i, sphere_list_lengths[i]);
    }
    printf("DEBUG: output of theta_dot * W\n");
    fflush(stdout);
    FILE* masked_theta_dot_out = fopen(create_output_filepath("masked_theta_dot_out.dat"), "w");
    for(int i = 0; i < N3; i++) {
        fprintf(masked_theta_dot_out, "%.15e\n", creal(theta_dot[i]));
    }
    fclose(masked_theta_dot_out);
    printf("DEBUG: output of FFT(theta_dot * W) \n");
    fflush(stdout);
    FILE* masked_theta_dot_fft_out = fopen(create_output_filepath("masked_theta_dot_fft_out.dat"), "w");
    for(int i = 0; i < N3; i++) {
        const double Re = creal(theta_dot_fft[i]);
        const double Im = cimag(theta_dot_fft[i]);
        fprintf(masked_theta_dot_fft_out, "%.15e+%.15ej\n", Re, Im);
    }
    fclose(masked_theta_dot_fft_out);
#endif


    // spectrum of W*dot theta
    // P_field(k) = k^2 / L^3 \int d \Omega / 4\pi 0.5 * | field(k) |^2
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < NBINS; i++) {
        spectrum_uncorrected[i] = 0.0;
        const double bin_k = i * bin_width + bin_width/2.0;
        for(int j = 0; j < sphere_list_lengths[i]; j++) {
            struct Index index = spheres[i][j];
            const double integrant = theta_dot_fft[AT(index.ix, index.iy, index.iz)];
            spectrum_uncorrected[i] += creal(integrant)*creal(integrant) + cimag(integrant)*cimag(integrant);
        }
        spectrum_uncorrected[i] *= surface_integral_element[i];
        spectrum_uncorrected[i] *= bin_k*bin_k / (L*L*L) / (4 * PI) * 0.5;
    }

    // W fft
    fftw_execute(W_fft_plan);

    // compute M
    // M = 1 / (L^3)^2 * \int d \Omega / 4\pi d \Omega' / 4\pi |W(\vec{k} - \vec{k}')|^2
    // NOTE: this is the computationally most expensive part!
    const double f = pow(L, 6) * pow(4 * PI, 2);
    for(int i = 0; i < NBINS; i++) {
        for(int j = i; j < NBINS; j++) {
#ifdef DEBUG
            if(j % (NBINS/4) == 0) {
                printf("DEBUG: integrating M[%i, %i] of %ix%i\n", i, j, NBINS, NBINS);
            }
#endif
            gsl_matrix_set(M, i, j, i == j);
            gsl_matrix_set(M, j, i, i == j);
            continue;
            // integrate spheres
            double s = 0.0;
            #pragma omp parallel for collapse(2) reduction(+:s)
            for(int n1 = 0; n1 < sphere_list_lengths[i]; n1++) {
                for(int n2 = 0; n2 < sphere_list_lengths[j]; n2++) {
                    const struct Index idx1 = spheres[i][n1];
                    const struct Index idx2 = spheres[j][n2];
                    const int ix = substract_wave_numbers(idx1.ix, idx2.ix);
                    const int iy = substract_wave_numbers(idx1.iy, idx2.iy);
                    const int iz = substract_wave_numbers(idx1.iz, idx2.iz);
                    const complex double f = W_fft[AT(ix, iy, iz)];
                    const double Re = creal(f);
                    const double Im = cimag(f);
                    s += Re*Re + Im*Im;
                }
            }
            s *= surface_integral_element[i] * surface_integral_element[j] / f;
            gsl_matrix_set(M, i, j, s);
            gsl_matrix_set(M, j, i, s);
        }
    }

    // FILE* m_out = fopen(create_output_filepath("debug_M.dat"), "w");
    // gsl_matrix_fprintf(m_out, M, "%f");
    // fclose(m_out);

    // invert M
    // definition of M^-1:
    // \int k'^2 dk' / 2\pi^2 M^{-1}(k, k') M(k', k'') = 2\pi^2/k^2 \delta(k - k'')
    //
    // bin_width * sum_{k'} k'^2 / (2*np.pi^2) * M^-1(k,k') M(k',k'') = 2pi^2/k^2 delta(k - k'')
    // tilde M^-1 = bin_with / (2pi^2)^2 * k^2 k'^2 * M^-1(k, k')
    // M^-1(k, k') = (2pi^2)^2 / (bin_with * k^2 k'^2) * tilde M^-1(k, k')
    // sum_k' tilde M^-1(k, k')  M(k', k'') = delta(k, k'')
    int sign;
    gsl_linalg_LU_decomp(M, p, &sign);
    gsl_linalg_LU_invert(M, p, M_inv);
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < NBINS; i++) {
        for(int j = i; j < NBINS; j++) {
            const double M_inv_elem = gsl_matrix_get(M_inv, i, j);
            const double bin_k_1 = i * bin_width + bin_width/2;
            const double bin_k_2 = j * bin_width + bin_width/2;
            gsl_matrix_set(M_inv, i, j,
                    M_inv_elem * pow(2 * PI, 2) / (bin_width * bin_k_1*bin_k_1 * bin_k_2*bin_k_2));
        }
    }

    // multiply spectrum by M
    #pragma omp parallel for
    for(int i = 0; i < NBINS; i++) {
        double s = 0.0;
        for(int j = 0; j < NBINS; j++) {
            s += gsl_matrix_get(M_inv, i, j) * spectrum_uncorrected[j];
        }
        spectrum_corrected[i] = s;
        const double bin_k = i * bin_width + bin_width/2;
        const double f = pow(bin_k, 2) / pow(L, 3) / (2*PI*PI) * bin_width;
        spectrum_corrected[i] *= f;
    }

    // output spectrum
    for(int i = 0; i < NBINS; i++) {
        const double bin_k = i * bin_width + bin_width/2;
        fprintf(out, "%i %.15e %.15e %.15e\n", step, bin_k, spectrum_uncorrected[i], spectrum_corrected[i]);
    }
}

