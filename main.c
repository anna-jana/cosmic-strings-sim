#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#include "globals.h"

#define EVERY_ANALYSIS_STEP 10

int main(int argc, char* argv[]) {
    init();
    init_detect_strings();
    write_slice_xy("initial_slice.dat", 0);
    int saved = 0;
    for(step = 0; step < NSTEPS; step++) {
        printf("\rrunning simulation: step = %i/%i, saved = %i, conformal time = %lf, log = %lf",
                step + 1, NSTEPS, saved, current_conformal_time, TAU_TO_LOG(current_conformal_time));
        fflush(stdout);
        make_step();
        if(step % EVERY_ANALYSIS_STEP == 0) {
            compute_axion();
            detect_strings();
            saved++;
            break;
        }
    }
    printf("\n");
    write_slice_xy("final_slice.dat", 0);
    deinit();
    deinit_detect_strings();
    return EXIT_SUCCESS;
}

