#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#include "globals.h"

void write_slice(char* fname) {
    printf("writing slice to %s\n", fname);
    FILE* out = fopen(fname, "w");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            fprintf(out, "%lf+%lfj ", creal(phi[AT(i, j, 0)]), cimag(phi[AT(i, j, 0)]));
        }
        fprintf(out, "\n");
    }
    fclose(out);
}

int main(int argc, char* argv[]) {
    init();
    write_slice("initial_slice.dat");
    for(i = 0; i < NSTEPS; i++)
        step();
    printf("\n");
    write_slice("final_slice.dat");
    deinit();
    return EXIT_SUCCESS;
}

