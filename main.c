#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

#include "globals.h"

int main(int argc, char* argv[]) {
    init();
    init_detect_strings();
    init_energy_computation();

    FILE* out = fopen(PARAMETER_FILENAME, "w");
    fprintf(out, "{\n");
    fprintf(out, "\"L\": %lf,\n", L);
    fprintf(out, "\"LOG_START\": %lf,\n", LOG_START);
    fprintf(out, "\"LOG_END\": %lf,\n", LOG_END);
    fprintf(out, "\"N\": %i,\n", N);
    fprintf(out, "\"DELTA\": %lf\n", DELTA);
    fprintf(out, "}\n");
    fclose(out);

    for(step = 0; step < NSTEPS; step++) {
        double l = TAU_TO_LOG(current_conformal_time);
        double H = 1/exp(l);
        printf("\rINFO: running simulation: step = %i/%i, conformal time = %lf, log = %lf, H = %lf",
                step + 1, NSTEPS, current_conformal_time, l, H);
        fflush(stdout);
        make_step();
        if(step % EVERY_ANALYSIS_STEP == 0 || step == NSTEPS - 1) {
            printf("\nINFO: performing analysis: string detection, energy components");
            fflush(stdout);
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

