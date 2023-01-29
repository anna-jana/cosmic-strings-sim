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

struct Index {
    int ix, iy, iz;
};

int points_capacity;
int points_length;
struct Index* points;

FILE* out_strings;
char* string_fname = "strings.dat";

// external interface
void init_detect_strings(void) {
    printf("INFO: writing strings to %s\n", string_fname);
    out_strings = fopen(string_fname, "w");

    points_capacity = 2*N;
    points_length = 0;
    points = malloc(sizeof(struct Index) * points_capacity);
}

void deinit_detect_strings(void) {
    fclose(out_strings);
    free(points);
}

static void group_strings(void);
static void find_string_points(void);

void detect_strings(void) {
    find_string_points();
    group_strings();
}

// list of points
static inline void clear_points(void) {
    points_length = 0;
}

static inline void add_point(struct Index p) {
    if(points_length >= points_capacity) {
        points_capacity *= 2;
        points = realloc(points, sizeof(struct Index) * points_capacity);
    }
    points[points_length++] = p;
}

static inline void remove_point(int i) {
    points[i] = points[--points_length];
}

static inline struct Index pop_point(void) {
    return points[--points_length];
}

// detect string in placet
static inline bool crosses_real_axis(complex double phi1, complex double phi2) {
    return cimag(phi1) * cimag(phi2) < 0;
}

static inline int handedness(complex double phi1, complex double phi2) {
    return sign(cimag(phi1 * conj(phi2)));
}

static inline bool loop_contains_string(complex double phi1, complex double phi2,
                                   complex double phi3, complex double phi4) {
    int loop = (
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
    clear_points();
    for(int ix = 0; ix < N; ix++) {
        for(int iy = 0; iy < N; iy++) {
            for(int iz = 0; iz < N; iz++) {
                bool contains_string = is_string_at_xy(ix, iy, iz) ||
                                       is_string_at_yz(ix, iy, iz) ||
                                       is_string_at_zx(ix, iy, iz);
                if(contains_string) {
                    struct Index idx = {ix, iy, iz};
                    add_point(idx);
                }
            }
        }
    }
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
    int current_string_index = 0;
    while(points_length > 0) {
        struct Index initial_point = pop_point();
        struct Index last_point = initial_point;
        int current_string_length = 1;
        while(true) {
            if(points_length == 0) {
                if(current_string_length >= 2) {
                    double dist_beginning = cyclic_dist_squared(
                            initial_point, last_point);
                    if(dist_beginning <= MAXIMAL_DISTANCE) {
                        fprintf(out_strings, "%i %i %i %i %i\n", step, current_string_index,
                                initial_point.ix, initial_point.iy, initial_point.iz);
                        current_string_length++;
                    }
                }
            }
            double min_d = 1.0/0.0;
            int min_i = -1;
            for(int i = 0; i < points_length; i++) {
                double d = cyclic_dist_squared(points[i], last_point);
                if(d < min_d) {
                    min_d = d;
                    min_i = i;
                }
            }
            if(current_string_length >= MIN_STRING_LENGTH) {
                double dist_beginning = cyclic_dist_squared(
                        initial_point, points[min_i]);
                if(dist_beginning < min_d) break;
            }
            if(min_d > MAXIMAL_DISTANCE) break;
            // output
            last_point = points[min_i];
            fprintf(out_strings, "%i %i %i %i %i\n", step, current_string_index,
                    last_point.ix, last_point.iy, last_point.iz);
            current_string_length++;
            remove_point(min_i);
        }
        current_string_index++;
    }
}

