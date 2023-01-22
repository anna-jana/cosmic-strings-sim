#include <stdlib.h>

#include "globals.h"

/*************************************** math functions ********************************/
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

int mod(int a, int b) {
    return ((a % b) + b) % b;
}

int sign(double x) {
    if(x > 0.0) return 1;
    if(x < 0.0) return -1;
    return 0;
}

void write_slice_xy(char* fname, int iz) {
    printf("INFO: writing slice to %s\n", fname);
    FILE* out = fopen(fname, "w");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            fprintf(out, "%lf+%lfj ", creal(phi[AT(i, j, iz)]), cimag(phi[AT(i, j, iz)]));
        }
        fprintf(out, "\n");
    }
    fclose(out);
}

void write_field(char* fname) {
    printf("\nINFO: writing grid to %s", fname);
    FILE* out = fopen(fname, "w");
    for(int iz = 0; iz < N; iz++) {
        for(int ix = 0; ix < N; ix++) {
            for(int iy = 0; iy < N; iy++) {
                fprintf(out, "%lf+%lfj ", creal(phi[AT(ix, iy, iz)]), cimag(phi[AT(ix, iy, iz)]));
            }
            fprintf(out, "\n");
        }
    }
    fclose(out);
}

