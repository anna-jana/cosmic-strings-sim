#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "globals.h"

int main(void) {
    create_output_dir();
    output_parameters();

    init();
    init_detect_strings();
    init_energy_computation();
    init_compute_spectrum();

    write_field("initial_field.dat", phi);

    for(step = 0; step < NSTEPS; step++) {
        const double l = TAU_TO_LOG(current_conformal_time);
        const double H = 1/exp(l);
        printf("\rINFO: running simulation: step = %i/%i, conformal time = %lf, log = %lf, H = %lf",
                step + 1, NSTEPS, current_conformal_time, l, H);
        fflush(stdout);

        make_step();

        if(step % EVERY_ANALYSIS_STEP == 0 || step == NSTEPS - 1) {
            printf("\nINFO: performing analysis: string detection, energy components, axion spectrum");
            fflush(stdout);
            detect_strings();
            compute_energy();
            compute_spectrum();
        }
    }

    write_field("final_field.dat", phi);
    write_field("final_field_dot.dat", phi_dot);
    printf("\n");

    deinit();
    deinit_detect_strings();
    deinit_energy_computation();
    deinit_compute_spectrum();

    return EXIT_SUCCESS;
}
