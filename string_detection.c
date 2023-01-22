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

struct Index {
    int patch;
    int ix, iy, iz;
};

bool* close_to_string;
int* patch;
FILE* strings_out;
int not_expanded_list_capacity;
int not_expanded_list_length;
struct Index* not_expanded_list;
char* string_fname = "strings.dat";

void init_detect_strings(void) {
    printf("INFO: writing strings to %s\n", string_fname);
    strings_out = fopen(string_fname, "w");
    close_to_string = malloc(sizeof(bool) * N3);
    patch = malloc(sizeof(int) * N3);
    not_expanded_list_capacity = 10;
    not_expanded_list_length = 0;
    not_expanded_list = malloc(sizeof(struct Index) * not_expanded_list_capacity);
}

void deinit_detect_strings(void) {
    fclose(strings_out);
    free(close_to_string);
    free(patch);
    free(not_expanded_list);
}

void add_to_not_expanded_list(struct Index p) {
    if(not_expanded_list_length >= not_expanded_list_capacity) {
        not_expanded_list_capacity *= 2;
        not_expanded_list = realloc(not_expanded_list,
                sizeof(struct Index) * not_expanded_list_capacity);
    }
    not_expanded_list[not_expanded_list_length++] = p;
}

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

void detect_strings(void) {
    // find grid points close to strings
    for(int ix = 0; ix < N; ix++) {
        for(int iy = 0; iy < N; iy++) {
            for(int iz = 0; iz < N; iz++) {
                close_to_string[AT(ix, iy, iz)] = is_string_at_xy(ix, iy, iz) ||
                                                  is_string_at_yz(ix, iy, iz) ||
                                                  is_string_at_zx(ix, iy, iz);
            }
        }
    }

    // group them in the xy plane
#ifdef DEBUG
    char* patches_fname = "patches.dat";
    FILE* patches_out = fopen(patches_fname, "w");
    printf("\nINFO: writing patches to %s", patches_fname);
#endif
    memset(patch, 0, sizeof(int) * N3);
    int patch_count = 1;
    // TODO: this is not the cache efficient loop order
    for(int iz = 0; iz < N; iz++) {
        // search for next not looked at point which is marked is_close
        for(int ix = 0; ix < N; ix++) {
            for(int iy = 0; iy < N; iy++) {
                if(close_to_string[AT(ix, iy, iz)] && patch[AT(ix, iy, iz)] == 0) {
                    int patch_id = patch_count++;
                    // add to not expanded list
                    struct Index p = {patch_id, ix, iy, iz};
                    add_to_not_expanded_list(p);
                    // while not expanded list is not empty:
                    while(not_expanded_list_length > 0) {
                        // take point from not expanded list
                        struct Index p = not_expanded_list[--not_expanded_list_length];
                        patch[AT(p.ix, p.iy, p.iz)] = patch_id;
                        // patch != 0 then the neighbor is already in some patch_list
                        // expand neightbors and add to not expanded list
                        // if not marked or not_expanded yet
                        for(int dix = -1; dix < 2; dix++) {
                            for(int diy = -1; diy < 2; diy++) {
                                // neighboring point to p
                                struct Index q = {
                                    patch_id,
                                    mod(p.ix + dix, N),
                                    mod(p.iy + diy, N),
                                    p.iz
                                };
                                if(!(dix == 0 && diy == 0) && close_to_string[AT(q.ix, q.iy, q.iz)]
                                        && patch[AT(q.ix, q.iy, q.iz)] == 0) {
                                    // check if q is already in not_expanded
                                    bool in_not_expanded = false;
                                    for(int i = 0; i < not_expanded_list_length; i++) {
                                        if(not_expanded_list[i].ix == q.ix &&
                                           not_expanded_list[i].iy == q.iy &&
                                           not_expanded_list[i].iz == q.iz) {
                                            in_not_expanded = true;
                                            break;
                                        }
                                    }
                                    // add to not_expanded if not already in not_expanded
                                    if(!in_not_expanded) {
                                        add_to_not_expanded_list(q);
                                    }
                                }
                            }
                        }
                    }
                }
#ifdef DEBUG
                fprintf(patches_out, "%i ", patch[AT(ix, iy, iz)]);
#endif
            }
#ifdef DEBUG
            fprintf(patches_out, "\n");
#endif
        }
    }
#ifdef DEBUG
    fclose(patches_out);
#endif

    // compute centers of patches and their connections
    int sum_x[patch_count];
    int sum_y[patch_count];
    int patch_iz[patch_count];
    int count[patch_count];
    int connected[patch_count];
#ifdef DEBUG
    int nconnections = 0, nbad_connections = 0;
#endif
    for(int i = 1; i < patch_count; i++)
        sum_x[i] = sum_y[i] = count[i] = connected[i] = 0;
    for(int iz = 0; iz < N; iz++) {
        for(int ix = 0; ix < N; ix++) {
            for(int iy = 0; iy < N; iy++) {
                int p = patch[AT(ix, iy, iz)];
                if(p != 0) {
                    sum_x[p] += ix;
                    sum_y[p] += iy;
                    count[p]++;
                    patch_iz[p] = iz;
                    int q;
                    q = patch[AT(ix, iy, mod(iz - 1, N))];
                    if(q != 0) {
                        if(connected[p] == 0 || connected[p] == q) {
                            connected[p] = q;
#ifdef DEBUG
                            nconnections++;
#endif
                        } else {
#ifdef DEBUG
                            nbad_connections++;
#endif
                        }
                    }
                }
            }
        }
    }
#ifdef DEBUG
    printf("\nDEBUG: nconnections: %i, nbad_connections: %i", nconnections, nbad_connections);
#endif

    int string_index = 0;
    int remaining_patches[patch_count - 1];
    for(int i = 0; i < patch_count; i++)
        remaining_patches[i] = i + 1;
    int remaining_patches_length = patch_count - 1;
    while(remaining_patches_length > 0) {
        int current_patch = remaining_patches[--remaining_patches_length];
        int first_patch_of_string = current_patch;
        while(true) {
            double x = dx * sum_x[current_patch] / (double) count[current_patch];
            double y = dx * sum_y[current_patch] / (double) count[current_patch];
            double z = patch_iz[current_patch] * dx;
            fprintf(strings_out, "%i %i %i %i %lf %lf %lf\n",
                    step, string_index, current_patch, connected[current_patch], x, y, z);
            int next = connected[current_patch];
            if(next == first_patch_of_string || next == 0)
                break;
            current_patch = next;
            for(int i = 0; i < remaining_patches_length; i++) {
                if(remaining_patches[i] == next) {
                    remaining_patches[i] = remaining_patches[--remaining_patches_length];
                }
            }

        }
        string_index++;
    }
}
