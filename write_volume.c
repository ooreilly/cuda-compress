#include <stdio.h>
#include <stdlib.h>
#include "write_volume.h"
#include "init_random.h"
#include "init_x.h"
#include "init_y.h"
#include "init_z.h"



int main(int argc, char **argv) {

        
        const char *filename = argv[1];
        int nx = atoi(argv[2]);
        int ny = atoi(argv[3]);
        int nz = atoi(argv[4]);
        int bx = atoi(argv[5]);
        int by = atoi(argv[6]);
        int bz = atoi(argv[7]);
        int grid = atoi(argv[8]);

        float *x;
        switch (grid) {
                case 0:
                printf("Generating x-grid \n");
                init_x(x, nx, ny, nz, bx, by, bz);
                break;
                case 1:
                printf("Generating y-grid \n");
                init_y(x, nx, ny, nz, bx, by, bz);
                break;
                case 2:
                printf("Generating z-grid \n");
                init_z(x, nx, ny, nz, bx, by, bz);
                break;
                case 3:
                printf("Generating random grid \n");
                init_random(x, nx, ny, nz, bx, by, bz);
                break;
        }

        if (argc != 9) {
                printf("usage: %s <filename> <nx> <ny> <nz> <bx> <by> <bz> <grid>\n" \
                " grid :  Type of grid to generate, x(0), y(1), z(2), or random(3) \n", 
                argv[0]);
                return -1;
        }

        
        printf("Writing %dx%dx%d blocks of dimension %dx%dx%d to %s \n", bx, by, bz, nx, ny, nz, 
                filename);
        write_volume(filename, x, nx, ny, nz, bx, by, bz);
        free(x);
        
}
