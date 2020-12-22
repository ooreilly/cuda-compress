#include <stdio.h>
#include <stdlib.h>
#include "write_volume.h"
#include "init_x.h"
#include "init_z.h"



int main(int argc, char **argv) {

        
        const char *filename = argv[1];
        int nx = atoi(argv[2]);
        int ny = atoi(argv[3]);
        int nz = atoi(argv[4]);
        int bx = atoi(argv[5]);
        int by = atoi(argv[6]);
        int bz = atoi(argv[7]);

        float *x;
        init_x(x, nx, ny, nz, bx, by, bz);

        if (argc != 8) {
                printf("usage: %s <filename> <nx> <ny> <nz> <bx> <by> <bz>\n", argv[0]);
                return -1;
        }

        
        printf("Writing %dx%dx%d blocks of dimension %dx%dx%d to %s \n", bx, by, bz, nx, ny, nz, 
                filename);
        write_volume(filename, x, nx, ny, nz, bx, by, bz);
        free(x);
        
}
