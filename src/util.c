#include <stdlib.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <assert.h>
#include <stdbool.h>

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
        return 2 * PI * (n / 2) / (d*n);
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

void write_field(char* fname, const complex double* field) {
    char* fpath = create_output_filepath(fname);
    printf("\nINFO: writing grid to %s\n", fpath);
    FILE* out = fopen(fpath, "w");
    for(int iz = 0; iz < N; iz++) {
        for(int iy = 0; iy < N; iy++) {
            for(int ix = 0; ix < N; ix++) {
                fprintf(out, "%lf+%lfj ",
                        creal(field[AT(ix, iy, iz)]),
                        cimag(field[AT(ix, iy, iz)]));
            }
            fprintf(out, "\n");
        }
    }
    fclose(out);
}

void output_parameters(void) {
    char* param_fpath = create_output_filepath(PARAMETER_FILENAME);
    printf("INFO: writing parameters to %s\n", param_fpath);
    FILE* out = fopen(param_fpath, "w");
    fprintf(out, "{\n");
    fprintf(out, "\"L\": %lf,\n", L);
    fprintf(out, "\"LOG_START\": %lf,\n", LOG_START);
    fprintf(out, "\"LOG_END\": %lf,\n", LOG_END);
    fprintf(out, "\"N\": %i,\n", N);
    fprintf(out, "\"DELTA\": %lf\n", DELTA);
    fprintf(out, "}\n");
    fclose(out);
}

#define MAX_PATH_SIZE 1024

static char output_dir[MAX_PATH_SIZE];
static char filepath_buffer[MAX_PATH_SIZE + MAX_PATH_SIZE + 1];

void create_output_dir(void) {
    int i = 1;
    while(true) {
        sprintf(output_dir, "run%i_output", i);
        DIR* dir = opendir(output_dir);
        if (dir) {
            closedir(dir);
            i++;
            continue;
        }
        assert(ENOENT == errno);
        mkdir(output_dir, S_IRWXU);
        printf("INFO: output directory is %s\n", output_dir);
        return;
    }
}

char* create_output_filepath(const char* filename) {
    sprintf(filepath_buffer, "%s/%s", output_dir, filename);
    return filepath_buffer;
}
