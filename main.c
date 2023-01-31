#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#include "globals.h"

int main(int argc, char* argv[]) {
    init();
    init_detect_strings();
    init_energy_computation();
    for(step = 0; step < NSTEPS; step++) {
        printf("\rINFO: running simulation: step = %i/%i, conformal time = %lf, log = %lf",
                step + 1, NSTEPS, current_conformal_time, TAU_TO_LOG(current_conformal_time));
        fflush(stdout);
        make_step();
        if(step % EVERY_ANALYSIS_STEP == 0 || step == NSTEPS - 1) {
            detect_strings();
            compute_energy();
        }
    }
    write_field("final_field.dat");
    printf("\n");
    deinit();
    deinit_detect_strings();
    deinit_energy_computation();
    return EXIT_SUCCESS;
}

