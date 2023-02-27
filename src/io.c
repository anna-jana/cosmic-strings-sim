#include <stdlib.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "globals.h"

void write_field(char* fname, const complex double* field) {
    char* fpath = create_output_filepath(fname);
    printf("INFO: writing grid to %s\n", fpath);
    FILE* out = fopen(fpath, "w");
    for(int iz = 0; iz < N; iz++) {
        for(int iy = 0; iy < N; iy++) {
            for(int ix = 0; ix < N; ix++) {
                fprintf(out, "%.15e+%.15ej ",
                        creal(field[AT(ix, iy, iz)]),
                        cimag(field[AT(ix, iy, iz)]));
            }
            fprintf(out, "\n");
        }
    }
    fclose(out);
}

#define PARAMETER_FILENAME "parameter.json"
void output_parameters(void) {
    char* param_fpath = create_output_filepath(PARAMETER_FILENAME);
    printf("INFO: writing parameters to %s\n", param_fpath);
    FILE* out = fopen(param_fpath, "w");
    const double final_tau = TAU_START + NSTEPS * Delta_tau;
    const double final_log = TAU_TO_LOG(final_tau);
    fprintf(out, "{\n");
    fprintf(out, "\"L\": %.15e,\n", L);
    fprintf(out, "\"LOG_START\": %.15e,\n", LOG_START);
    fprintf(out, "\"LOG_END\": %.15e,\n", LOG_END);
    fprintf(out, "\"LOG_FINAL\": %.15e,\n", final_log);
    fprintf(out, "\"TAU_FINAL\": %.15e,\n", final_tau);
    fprintf(out, "\"N\": %i,\n", N);
    fprintf(out, "\"Delta_tau\": %.15e,\n", Delta_tau);
    fprintf(out, "\"SEED\": %i\n", SEED);
    fprintf(out, "}\n");
    fclose(out);
}

#define MAX_PATH_SIZE 1024

static char output_dir[MAX_PATH_SIZE];
static char filepath_buffer[MAX_PATH_SIZE + MAX_PATH_SIZE + 1];

void create_output_dir(void) {
    int i = 1;
    while(true) {
        sprintf(output_dir, "run%i_output", i);
        DIR* dir = opendir(output_dir);
        if (dir) {
            closedir(dir);
            i++;
            continue;
        }
        assert(ENOENT == errno);
        mkdir(output_dir, S_IRWXU);
        printf("INFO: output directory is %s\n", output_dir);
        return;
    }
}

char* create_output_filepath(const char* filename) {
    sprintf(filepath_buffer, "%s/%s", output_dir, filename);
    return filepath_buffer;
}

bool parse_arg(int argc, char* argv[], char* arg, char type, bool required, void* value) {
    int i;
    bool found = false;
    for(i = 0; i < argc; i++) {
        if(strcmp(argv[i], arg) == 0) {
            found = true;
            break;
        }
    }
    if(!found) {
        if(required) {
            fprintf(stderr, "did not found option %s and its required\n", arg);
            exit(-1);
        }
        return false;
    }
    if(i < argc - 1) {
        int ok;
        if(type == 'f') {
            ok = sscanf(argv[i + 1], "%lf", (double*) value);
        } else if(type == 'i') {
            ok = sscanf(argv[i + 1], "%i", (int*) value);
        } else {
            fprintf(stderr, "invalid type specifier %c\n", type);
            exit(-1);
        }
        if(ok != 1) {
            fprintf(stderr, "expected number after option %s but found %s\n",
                    arg, argv[i + 1]);
            exit(-1);
        }
    } else {
        fprintf(stderr,
                "expected argument after option %s but it was the last argument\n",
                arg);
        exit(-1);
    }
    return true;
}

