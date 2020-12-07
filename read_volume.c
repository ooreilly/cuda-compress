#include <stdio.h>
#include <stdlib.h>

#include "printing.cuh"
#include "read_volume.h"



int main(int argc, char **argv) {

        
        const char *filename = argv[1];

        int x0 = 0, y0 = 0, z0 = 0;

        if (argc == 5) {
                x0 = atoi(argv[2]);
                y0 = atoi(argv[3]);
                z0 = atoi(argv[4]);
        }

        float *x;
        printf("reading: %s \n", filename);
        int nx, ny, nz, bx, by, bz;
        read_volume(filename, x, nx, ny, nz, bx, by, bz);
        if (argc == 5) {
                printf("Displaying block: %d %d %d \n", x0, y0, z0);
                print_array(&x[nx * ny * nz * (x0 + nx * (y0 + ny * z0))], nx, ny, nz);
        }

        free(x);
        
}

