#include "globals.h"

#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <complex.h>

struct Index {
    int patch;
    int ix, iy, iz;
};

double *theta;
bool* close_to_string;
int* patch;
FILE* strings_out;

void init_detect_strings(void) {
    theta = malloc(sizeof(double) * N3);
    strings_out = fopen("strings.dat", "w");
    close_to_string = malloc(sizeof(bool) * N3);
    patch = malloc(sizeof(int) * N3);
}

void deinit_detect_strings(void) {
    fclose(strings_out);
    free(theta);
    free(close_to_string);
    free(patch);
}

void compute_axion(void) {
    for(int i = 0; i < N3; i++)
        theta[i] = carg(phi[i]);
    FILE* out = fopen("axion.dat", "w");
    for(int ix = 0; ix < N; ix++) {
        for(int iy = 0; iy < N; iy++) {
            for(int iz = 0; iz < N; iz++) {
                fprintf(out, "%lf ", theta[AT(ix, iy, iz)]);
            }
            fprintf(out, "\n");
        }
    }
}

void detect_strings(void) {
    // find grid points close to strings
    FILE* is_close_out = fopen("is_close.dat", "w");
    for(int ix = 0; ix < N; ix++) {
        for(int iy = 0; iy < N; iy++) {
            for(int iz = 0; iz < N; iz++) {
                close_to_string[AT(ix, iy, iz)] =
                   abs(theta[AT(ix, iy, iz)] - theta[CYCLIC_AT(ix + 1, iy, iz)]) > PI/2 ||
                   abs(theta[AT(ix, iy, iz)] - theta[CYCLIC_AT(ix - 1, iy, iz)]) > PI/2 ||
                   abs(theta[AT(ix, iy, iz)] - theta[CYCLIC_AT(ix, iy + 1, iz)]) > PI/2 ||
                   abs(theta[AT(ix, iy, iz)] - theta[CYCLIC_AT(ix, iy - 1, iz)]) > PI/2 ||
                   abs(theta[AT(ix, iy, iz)] - theta[CYCLIC_AT(ix, iy, iz + 1)]) > PI/2 ||
                   abs(theta[AT(ix, iy, iz)] - theta[CYCLIC_AT(ix, iy, iz - 1)]) > PI/2;
                fprintf(is_close_out, "%i ", close_to_string[AT(ix, iy, iz)]);
            }
            fprintf(is_close_out, "\n");
        }
    }
    fclose(is_close_out);

    // group them in the xy plane
    FILE* patches_out = fopen("patches.dat", "w");
    memset(patch, 0, sizeof(int) * N3);
    int patch_count = 1;

    for(int iz = 0; iz < N; iz++) {
        // TODO: maybe move to global
        int not_expanded_list_capacity = 10;
        int not_expanded_list_length = 0;
        struct Index* not_expanded_list = malloc(sizeof(struct Index) * not_expanded_list_capacity);

        // search for next not looked at point which is marked is_close
        for(int ix = 0; ix < N; ix++) {
            for(int iy = 0; iy < N; iy++) {
                if(close_to_string[AT(ix, iy, iz)] && patch[AT(ix, iy, iz)] == 0) {
                    int patch_id = patch_count++;
                    // add to not expanded list
                    if(not_expanded_list_length >= not_expanded_list_capacity) {
                        not_expanded_list_capacity *= 2;
                        not_expanded_list = realloc(not_expanded_list, sizeof(struct Index) * not_expanded_list_capacity);
                    }
                    struct Index p = {patch_id, ix, iy, iz};
                    not_expanded_list[not_expanded_list_length++] = p;
                    // while not expanded list is not empty:
                    while(not_expanded_list_length > 0) {
                        // take point from not expanded list
                        struct Index p = not_expanded_list[--not_expanded_list_length];
                        patch[AT(p.ix, p.iy, p.iz)] = patch_id;
                                // patch != 0 then the neighbor is already in some patch_list
                                // patch != 0 then the neighbor is already in some patch_list
                        // expand neightbors and add to not expanded list if not marked or not_expanded yet
                        for(int dix = -1; dix < 2; dix++) {
                            for(int diy = -1; diy < 2; diy++) {
                                // neighboring point to p
                                struct Index q = {patch_id, mod(p.ix + dix, N), mod(p.iy + diy, N), p.iz};
                                if(!(dix == 0 && diy == 0) && close_to_string[AT(q.ix, q.iy, q.iz)] && patch[AT(q.ix, q.iy, q.iz)] == 0) {
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
                                        if(not_expanded_list_length >= not_expanded_list_capacity) {
                                            not_expanded_list_capacity *= 2;
                                            not_expanded_list = realloc(not_expanded_list,
                                                    sizeof(struct Index) * not_expanded_list_capacity);
                                        }
                                        not_expanded_list[not_expanded_list_length++] = q;
                                    }
                                }
                            }
                        }
                    }
                }
                fprintf(patches_out, "%i ", patch[AT(ix, iy, iz)]);
            }
            fprintf(patches_out, "\n");
        }
    }
    fclose(patches_out);

    int sum_x[patch_count];
    int sum_y[patch_count];
    int patch_iz[patch_count];
    int count[patch_count];
    int connected[patch_count];
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
                        } else {

                            printf("error: %i already connected to: %i but also connecst to %i\n",
                                    p, connected[p], q);
                        }
                    }
                }
            }
        }
    }
    for(int i = 1; i < patch_count; i++) {
        double x = sum_x[i] / (double) count[i];
        double y = sum_y[i] / (double) count[i];
        double z = patch_iz[i] * dx;
        fprintf(strings_out, "%lf %lf %lf %i %i\n",
                x, y, z, i, connected[i]);
    }

    //int string_index = 0;
    //int remaining_patches[patch_count - 1];
    //for(int i = 0; i < patch_count; i++)
    //    remaining_patches[i] = i + 1;
    //int remaining_patches_length = patch_count - 1;
    //while(remaining_patches_length > 0) {
    //    printf("remaining_patches_length: %i\n", remaining_patches_length);
    //    int current_patch = remaining_patches[--remaining_patches_length];
    //    int first_patch_of_string = current_patch;
    //    while(true) {
    //        printf("current_patch: %i\n", current_patch);
    //        double x = sum_x[current_patch] / (double) count[current_patch];
    //        double y = sum_y[current_patch] / (double) count[current_patch];
    //        double z = patch_iz[current_patch] * dx;
    //        fprintf(strings_out, "%i %i %lf %lf %lf\n", step, string_index, x, y, z);
    //        int next = to[current_patch];
    //        if(next == first_patch_of_string || next == 0)
    //            break;
    //        current_patch = next;
    //        for(int i = 0; i < remaining_patches_length; i++) {
    //            if(remaining_patches[i] == next) {
    //                remaining_patches[i] = remaining_patches[--remaining_patches_length];
    //            }
    //        }

    //    }
    //    string_index++;
    //}
}
