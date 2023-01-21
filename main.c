#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#include "globals.h"

#define EVERY_ANALYSIS_STEP 10

int main(int argc, char* argv[]) {
    init();
    init_detect_strings();
    write_slice_xy("initial_slice.dat", 0);
    for(step = 0; step < NSTEPS; step++) {
        printf("\rrunning simulation: step = %i/%i, conformal time = %lf, log = %lf",
                step + 1, NSTEPS, current_conformal_time, TAU_TO_LOG(current_conformal_time));
        fflush(stdout);
        make_step();
    }
    detect_strings();
    printf("\n");
    write_slice_xy("final_slice.dat", 0);
    deinit();
    deinit_detect_strings();
    return EXIT_SUCCESS;
}

