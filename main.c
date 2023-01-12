#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>

#define N 20
#define T 10
#define DELTA_T 1e-1
#define NSTEPS ((int)(ceil(T / DELTA_T)))
#define START_TIME 1.0

struct Field {
    complex double phi[N][N][N];
    complex double phi_dot[N][N][N];
};

struct SimulationState {
    double t;
    struct Field* current;
    struct Field* next;
};

struct Analysis {
};

void init(struct SimulationState* sim) {
}

void step(struct SimulationState* sim, double time) {
}

void analyse(struct Analysis* ana, struct SimulationState* sim, double time) {
}

int main(int argc, char* argv[]) {
    struct SimulationState sim;
    struct Analysis ana;
    printf("running simuation");
    init(&sim);
    for(int i = 0; i < NSTEPS; i++) {
        double time = START_TIME + i * DELTA_T;
        printf("step: %i, time: %lf\n", i, time);
        analyse(&ana, &sim, time);
        step(&sim, time);
    }
    printf("done");
    return EXIT_SUCCESS;
}

