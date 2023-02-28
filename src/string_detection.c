// mixture of the string detection algorithm from:
// the detection of strings within a loop:
// Moore: Axion dark matter: strings and their cores (appendix A.2)
// https://arxiv.org/pdf/1509.00026.pdf
// connection of the strings:
// Gorghetto: Axions from Strings: The Attractive Solution (appendix A.2)
// https://arxiv.org/pdf/1806.04677.pdf
// (yes its the same section)

#include "globals.h"

#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <complex.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

// thread save array list for string points
int* points_capacities;
int* points_lengths;
int* last_points_lengths;
struct Index** points;

static inline void clear_points(void) {
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {
        last_points_lengths[thread_id] = points_lengths[thread_id] = 0;
    }
}

static inline void add_point(struct Index p) {
    const int thread_id = omp_get_thread_num();
    if(points_lengths[thread_id] >= points_capacities[thread_id]) {
        points_capacities[thread_id] *= 2;
        points[thread_id] = realloc(points[thread_id],
                sizeof(struct Index) * points_capacities[thread_id]);
    }
    int i = points_lengths[thread_id];
    points[thread_id][i] = p;
    points_lengths[thread_id]++;
    last_points_lengths[thread_id]++;
}

static inline void remove_point(int thread_id, int i) {
    struct Index tmp = points[thread_id][i];
    points[thread_id][i] = points[thread_id][points_lengths[thread_id] - 1];
    points[thread_id][points_lengths[thread_id] - 1] = tmp;
    points_lengths[thread_id]--;
}

static inline struct Index pop_point(void) {
    for(int thread_id = num_threads - 1; thread_id >= 0; thread_id--) {
        if(points_lengths[thread_id] > 0) {
            return points[thread_id][--points_lengths[thread_id]];
        }
    }
    assert(false);
}

static inline bool are_points_empty(void) {
    for(int thread_id = 0; thread_id < num_threads; thread_id++) {
        if(points_lengths[thread_id] > 0)
            return false;
    }
    return true;
}

// file output
FILE* out_strings;
#define STRING_FNAME "strings.dat"

// external interface
void init_detect_strings(void) {
    const char* string_filepath = create_output_filepath(STRING_FNAME);
    printf("INFO: writing strings to %s\n", string_filepath);
    out_strings = fopen(string_filepath, "w");

    points_capacities = malloc(sizeof(int) * num_threads);
    for(int i = 0; i < num_threads; i++)
        points_capacities[i] = 2*N;

    points_lengths = malloc(sizeof(int) * num_threads);
    last_points_lengths = malloc(sizeof(int) * num_threads);
    clear_points();

    points = malloc(sizeof(struct Index*) * num_threads);
    for(int i = 0; i < num_threads; i++)
        points[i] = malloc(sizeof(struct Index) * points_capacities[i]);
}

void deinit_detect_strings(void) {
    fclose(out_strings);

    free(points_capacities);
    free(points_lengths);
    for(int i = 0; i < num_threads; i++)
        free(points[i]);
    free(points);
}

static void group_strings(void);
static void find_string_points(void);

void detect_strings(void) {
    find_string_points();
    group_strings();
}

// detect string in placet
static inline bool crosses_real_axis(complex double phi1, complex double phi2) {
    return cimag(phi1) * cimag(phi2) < 0;
}

static inline int handedness(complex double phi1, complex double phi2) {
    return sign(cimag(phi1 * conj(phi2)));
}

static inline bool loop_contains_string(complex double phi1, complex double phi2, complex double phi3, complex double phi4) {
    const int loop = (
        + crosses_real_axis(phi1, phi2) * handedness(phi1, phi2)
        + crosses_real_axis(phi2, phi3) * handedness(phi2, phi3)
        + crosses_real_axis(phi3, phi4) * handedness(phi3, phi4)
        + crosses_real_axis(phi4, phi1) * handedness(phi4, phi1)
    );
    return abs(loop) == 2;
}

static inline bool is_string_at_xy(int ix, int iy, int iz) {
    return loop_contains_string(phi[CYCLIC_AT(ix, iy, iz)], phi[CYCLIC_AT(ix + 1, iy, iz)],
                                phi[CYCLIC_AT(ix + 1, iy + 1, iz)], phi[CYCLIC_AT(ix, iy + 1, iz)]);
}

static inline bool is_string_at_yz(int ix, int iy, int iz) {
    return loop_contains_string(phi[CYCLIC_AT(ix, iy, iz)], phi[CYCLIC_AT(ix, iy + 1, iz)],
                                phi[CYCLIC_AT(ix, iy + 1, iz + 1)], phi[CYCLIC_AT(ix, iy, iz + 1)]);
}

static inline bool is_string_at_zx(int ix, int iy, int iz) {
    return loop_contains_string(phi[CYCLIC_AT(ix, iy, iz)], phi[CYCLIC_AT(ix, iy, iz + 1)],
                                phi[CYCLIC_AT(ix + 1, iy, iz + 1)], phi[CYCLIC_AT(ix + 1, iy, iz)]);
}

// find grid points close to strings
static void find_string_points(void) {
    const clock_t start_clock = clock();
    clear_points();
    #pragma omp parallel for collapse(3)
    for(int iz = 0; iz < N; iz++) {
        for(int iy = 0; iy < N; iy++) {
            for(int ix = 0; ix < N; ix++) {
                const bool contains_string = is_string_at_xy(ix, iy, iz) ||
                                       is_string_at_yz(ix, iy, iz) ||
                                       is_string_at_zx(ix, iy, iz);
                if(contains_string) {
                    const struct Index idx = {ix, iy, iz};
                    add_point(idx);
                }
            }
        }
    }
    const clock_t end_clock = clock();
    const double ms_elapsed = 1000.0 * (end_clock - start_clock) / (double) CLOCKS_PER_SEC;
#ifdef DEBUG
    printf("\nDEBUG: string points detection took: %lf ms\n", ms_elapsed);
    // sequential: ~133ms
#endif
}

#define MAXIMAL_DISTANCE (3*2*2)
#define MIN_STRING_LENGTH 3

inline static int cyclic_dist_squared_1d(int x1, int x2) {
    int d1 = x1 - x2;
    int d2 = N - x1 + x2;
    int d3 = N - x2 + x1;
    d1 *= d1;
    d2 *= d2;
    d3 *= d3;
    if(d1 > d2) {
        if(d2 > d3) {
            return d3;
        } else {
            return d2;
        }
    } else {
        if(d1 > d3) {
            return d3;
        } else {
            return d1;
        }
    }
}

inline static int cyclic_dist_squared(struct Index i, struct Index j) {
    return cyclic_dist_squared_1d(i.ix, j.ix) +
           cyclic_dist_squared_1d(i.iy, j.iy) +
           cyclic_dist_squared_1d(i.iz, j.iz);
}

// grouping points into strings
static void group_strings(void) {
    if(step == NSTEPS - 1) {
        int npoints = 0;
        for(int thread_id = 0; thread_id < num_threads; thread_id++) {
            printf("DEBUG: %i points on thread %i \n", points_lengths[thread_id], thread_id);
            npoints += points_lengths[thread_id];
        }
        printf("DEBUG: %i points found\n", npoints);
    }
    // find one string after the other
    int current_string_index = 0;
    while(!are_points_empty()) {
        // start with one point left in the points list
        const struct Index initial_point = pop_point();
        struct Index last_point = initial_point;
        fprintf(out_strings, "%i %i %i %i %i\n", step, current_string_index,
                last_point.ix, last_point.iy, last_point.iz);
        int current_string_length = 1;

        while(true) {
            // if the points are empty we check if we can connect to the beginning of
            // the current string
            if(are_points_empty()) {
                if(current_string_length >= 2) {
                    const double dist_beginning = cyclic_dist_squared(
                            initial_point, last_point);
                    if(dist_beginning <= MAXIMAL_DISTANCE) {
                        fprintf(out_strings, "%i %i %i %i %i\n", step, current_string_index,
                                initial_point.ix, initial_point.iy, initial_point.iz);
                        current_string_length++;
                    }
                    break;
                }
            }

            // search for the point left in the points list what is closest to the current point
            double min_d = 1.0/0.0;
            int min_i = -1;
            int min_thread_id = -1;
            for(int thread_id = 0; thread_id < num_threads; thread_id++) {
                for(int i = 0; i < points_lengths[thread_id]; i++) {
                    const struct Index p = points[thread_id][i];
                    const double d = cyclic_dist_squared(p, last_point);
                    if(d < min_d) {
                        min_d = d;
                        min_i = i;
                        min_thread_id = thread_id;
                    }
                }
            }

            // check if this point is actually closer then the beginning of
            // the current string
            if(current_string_length >= MIN_STRING_LENGTH) {
                const double dist_beginning = cyclic_dist_squared(
                        initial_point, points[min_thread_id][min_i]);
                // if beginning is closer -> we have a loop
                if(dist_beginning < min_d) break;
            }

            // check if the clostest point is too far away -> the have an open string here
            if(min_d > MAXIMAL_DISTANCE) break;

            // ok we dont have have and closed loop or the end of an open string ehre
            // -> just the next point on the string
            last_point = points[min_thread_id][min_i]; // the point found is now the newest found point
            fprintf(out_strings, "%i %i %i %i %i\n", step, current_string_index,
                    last_point.ix, last_point.iy, last_point.iz);
            current_string_length++;
            // remove from points left
            remove_point(min_thread_id, min_i);
        }
        current_string_index++;
    }
}

