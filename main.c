#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>

#include <fftw3.h>
// fftw3_complex is complex double bc
// complex.h is included before fftw3.h


#define PI 3.14159265358979323846
#define N 20
#define N3 (N*N*N)
#define T 10
#define DELTA_T 1e-1
#define NSTEPS ((int)(ceil(T / DELTA_T)))
#define START_TIME 1.0

// simulation state (global for now, maybe put into struct late)
double t;
int i;
fftw_complex *phi, *phi_dot;
fftw_complex *next_phi, *next_phi_dot;

void random_field(fftw_complex* field) {
}

void init(void) {
    printf("running simuation\n");
    t = START_TIME;
    i = 0;
    phi = fftw_malloc(sizeof(fftw_complex) * N3);
    phi_dot = fftw_malloc(sizeof(fftw_complex) * N3);
    next_phi = fftw_malloc(sizeof(fftw_complex) * N3);
    next_phi_dot = fftw_malloc(sizeof(fftw_complex) * N3);
    random_field(phi);
    random_field(phi_dot);
}

void deinit(void) {
    fftw_free(phi);
    fftw_free(phi_dot);
    fftw_free(next_phi);
    fftw_free(next_phi_dot);
    printf("done\n");
}


void step(void) {
    printf("step: %i, time: %lf\n", i, t);
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

double* fftwfreq(int n, double* data) {
    double* freq = malloc(sizeof(double) * n);
    double d = data[1] - data[0];
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

int main(int argc, char* argv[]) {
    init();
    for(i = 0; i < NSTEPS; i++) {
        step();
    }
    deinit();
    return EXIT_SUCCESS;
}

