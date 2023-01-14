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

// spacial discretisation
#define N 20
#define N3 (N*N*N)
#define AT(ix, iy, iz) ((ix) + (iy) * (N) + (iz) * (N) * (N))
#define L 1.0
// L/N not L/(N-1) bc we have cyclic boundary conditions
// *...*...* N = 2, dx = L / N
#define dx (L/N)

// time discretisation
#define T 10.0
#define DELTA_T 1e-1
#define NSTEPS ((int)(ceil(T / DELTA_T)))
#define START_TIME 1.0

// initial field generation
#define FIELD_MAX (1 / sqrt(2))
#define KMAX 0.3

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

double calc_k_max_grid(int n, double d) {
    if(n % 2 == 0) {
        return 2 * PI * (n / 2 - 1) / (d*n);
    } else {
        return 2 * PI * ((n - 1) / 2) / (d*n);
    }
}

double random_uniform(double min, double max) {
    return min + rand() / (double) RAND_MAX * (max - min);
}

// simulation state (global for now, maybe put into struct later)
double t;
int i;
fftw_complex *phi, *phi_dot;
fftw_complex *next_phi, *next_phi_dot;

// initial state generation
fftw_complex *hat;
double* ks;

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

void init(void) {
    printf("running simuation\n");
    assert(KMAX >= 0.0 && KMAX <= 1.0);

    t = START_TIME;
    i = 0;

    phi = fftw_malloc(sizeof(fftw_complex) * N3);
    phi_dot = fftw_malloc(sizeof(fftw_complex) * N3);
    next_phi = fftw_malloc(sizeof(fftw_complex) * N3);
    next_phi_dot = fftw_malloc(sizeof(fftw_complex) * N3);
    hat = fftw_malloc(sizeof(fftw_complex) * N3);

    ks = fft_freq(N, dx);
    random_field(phi);
    random_field(phi_dot);
}

void deinit(void) {
    fftw_free(phi);
    fftw_free(phi_dot);
    fftw_free(next_phi);
    fftw_free(next_phi_dot);
    fftw_free(hat);
    free(ks);
    printf("\ndone\n");
}


void step(void) {
    printf("\rstep: %i, time: %lf", i, t);
    t = START_TIME + i * DELTA_T;

    // propagate PDE
    // TODO

    fftw_complex* tmp;
    tmp = phi;
    phi = next_phi;
    next_phi = tmp;
    tmp = phi_dot;
    phi_dot = next_phi_dot;
    next_phi_dot = tmp;
}

int main(int argc, char* argv[]) {
    init();

    printf("writing slice\n");
    FILE* out = fopen("slice.dat", "w");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            fprintf(out, "%lf+%lfj ", creal(phi[AT(i, j, 0)]), cimag(phi[AT(i, j, 0)]));
        }
        fprintf(out, "\n");
    }
    fclose(out);

    for(i = 0; i < NSTEPS; i++) {
        step();
    }
    deinit();
    return EXIT_SUCCESS;
}

