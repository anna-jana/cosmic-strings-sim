#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>

#include <fftw3.h>

#define PI 3.14159265358979323846
#define n 400

int main(void) {
    // resourses
    double *freq = malloc(sizeof(double) * n);
    double *time = malloc(sizeof(double) * n);
    fftw_complex *time_dom = fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex *freq_dom = fftw_malloc(sizeof(fftw_complex) * n);

    // time domain range
    for(int i = 0; i < n; i++)
        time[i] = i / (double) (n - 1) * 2 * PI;

    // freq domain range
    double d = time[1] - time[0];
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

    // init time domain
    for(int i = 0; i < n; i++)
        time_dom[i] = (complex double) cos(time[i]);

    // transform to freq domain
    fftw_plan plan_forward = fftw_plan_dft_1d(n, time_dom, freq_dom, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan_forward);

    // take derivative in freq domain
    for(int i = 0; i < n; i++)
        freq_dom[i] *= freq[i] * I;

    // transform back to time domain (FFTW using unnormalized dft)
    fftw_plan plan_backward = fftw_plan_dft_1d(n, freq_dom, time_dom, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan_backward);
    for(int i = 0; i < n; i++)
        time_dom[i] /= n;

    // write output
    FILE* out = fopen("test.dat", "w");
    for(int i = 0; i < n; i++)
        fprintf(out, "%lf %lf %lf\n", time[i], cos(time[i]), creal(time_dom[i]));
    fclose(out);

    // release resourses
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
    fftw_free(time_dom);
    fftw_free(freq_dom);
    free(time);
    free(freq);

    return EXIT_SUCCESS;
}

