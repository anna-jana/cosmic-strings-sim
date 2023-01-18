#include "globals.h"

#include <stdbool.h>
#include <stdlib.h>
#include <complex.h>

double *theta;

struct StringPoint* strings;
int strings_len;
int strings_capacity;

// this implements an array list like c++ vectors or python lists for the string points
void add_string_point(struct StringPoint p) {
    if(strings_len > strings_capacity) {
        strings_capacity *= 2;
        strings = realloc(strings, sizeof(struct StringPoint) * strings_capacity);
    }
    strings[strings_len] = p;
    strings_len++;
}

void init_detect_strings(void) {
    theta = malloc(sizeof(double) * N3);
    strings_capacity = N;
    strings = malloc(sizeof(struct StringPoint) * strings_capacity);
    strings_len = 0;
}

void deinit_detect_strings(void) {
    free(theta);
    free(strings);
}

void compute_axion(void) {
    for(int i = 0; i < N3; i++)
        theta[i] = carg(phi[i]);
}

void detect_strings(void) {
    strings_len = 0; // clear string points buffer
    for(int ix = 0; ix < N; ix++) {
        for(int iy = 0; iy < N; iy++) {
            for(int iz = 0; iz < N; iz++) {
                // check loops of 2x2 grid point if they contain a string
                // the loops start at index (ix, iy, iz) and extend in positive direction
                // we check all directions i.e. i.e. xy, yz and zx planes
                // a loop contains a string if the difference of the theta angle
                // between too consecutive pointa on the loop is greater than PI/2
                // this method is used in the paper https://arxiv.org/pdf/1806.04677.pdf
                // see appendix A.2
                double d; // absolute difference in theta between to nodes an an 2x2 loop on the grid
                // ckeck xy plane
                d = fabs(theta[AT(ix, iy, iz)] - theta[AT(ix + 1, iy, iz)]);
                if(d > PI/2) goto add_string_xy; // add point and continue with yz plane
                d = fabs(theta[AT(ix + 1, iy, iz)] - theta[AT(ix + 1, iy + 1, iz)]);
                if(d > PI/2) goto add_string_xy; // add point and continue with yz plane
                d = fabs(theta[AT(ix + 1, iy + 1, iz)] - theta[AT(ix, iy + 1, iz)]);
                if(d > PI/2) goto add_string_xy; // add point and continue with yz plane
                d = fabs(theta[AT(ix, iy + 1, iz)] - theta[AT(ix, iy, iz)]);
                if(d > PI/2) goto add_string_xy; // add point and continue with yz plane
                goto yz_plane; // no strings point is to be added, continue with yz plane
                add_string_xy: { // add point in the xy plane
                    struct StringPoint p = {ix*dx + dx/2, iy*dx + dx/2, iz*dx};
                    add_string_point(p);
                }
                // ckeck yz plane
                yz_plane:
                d = fabs(theta[AT(ix, iy, iz)] - theta[AT(ix, iy + 1, iz)]);
                if(d > PI/2) goto add_string_yz; // add point and continue with zx plane
                d = fabs(theta[AT(ix, iy + 1, iz)] - theta[AT(ix, iy + 1, iz + 1)]);
                if(d > PI/2) goto add_string_yz; // add point and continue with zx plane
                d = fabs(theta[AT(ix, iy + 1, iz + 1)] - theta[AT(ix, iy, iz + 1)]);
                if(d > PI/2) goto add_string_yz; // add point and continue with zx plane
                d = fabs(theta[AT(ix, iy, iz + 1)] - theta[AT(ix, iy, iz)]);
                if(d > PI/2) goto add_string_yz; // add point and continue with zx plane
                goto zx_plane; // no string point is to be added, continue with zx plane
                add_string_yz: { // add point in the yz plane
                    struct StringPoint p = {ix*dx, iy*dx + dx/2, iz*dx + dx/2};
                    add_string_point(p);
                }
                // check zx plane
                zx_plane:
                d = fabs(theta[AT(ix, iy, iz)] - theta[AT(ix, iy, iz + 1)]);
                if(d > PI/2) goto add_string_zx; // add point and continue with next grid point
                d = fabs(theta[AT(ix, iy, iz + 1)] - theta[AT(ix + 1, iy, iz + 1)]);
                if(d > PI/2) goto add_string_zx; // add point and continue with next grid point
                d = fabs(theta[AT(ix + 1, iy, iz + 1)] - theta[AT(ix + 1, iy, iz)]);
                if(d > PI/2) goto add_string_zx; // add point and continue with next grid point
                d = fabs(theta[AT(ix + 1, iy, iz)] - theta[AT(ix, iy, iz)]);
                if(d > PI/2) goto add_string_zx; // add point and continue with next grid point
                continue; // next grid point
                add_string_zx: { // add point in the zx plane
                    struct StringPoint p = {ix*dx + dx/2, iy*dx, iz*dx + dx/2};
                    add_string_point(p);
                }
            }
        }
    }
}
