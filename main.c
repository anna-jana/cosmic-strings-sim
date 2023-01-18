#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#include "globals.h"

int main(int argc, char* argv[]) {
    init();
    init_detect_strings();
    write_slice_xy("initial_slice.dat", 0);
    for(i = 0; i < NSTEPS; i++) {
        step();
        compute_axion();
        detect_strings();
    }
    printf("\n");
    write_slice_xy("final_slice.dat", 0);
    deinit();
    deinit_detect_strings();
    return EXIT_SUCCESS;
}

