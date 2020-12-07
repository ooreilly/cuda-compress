#include <stdio.h>
#include <stdlib.h>
#include "write_volume.h"

void init_x(float*& out, int nx, int ny, int nz) {
        out = (float*)malloc(sizeof(float) * nx * ny * nz);
        for (int iz = 0; iz < nz; ++iz)
        for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix) {
                size_t pos = ix + nx * iy + nx * ny * iz;
                out[pos] = ix;// * iy * iz;      
        }
}


int main(int argc, char **argv) {

        
        const char *filename = argv[1];
        int nx = atoi(argv[2]);
        int ny = atoi(argv[3]);
        int nz = atoi(argv[4]);

        float *x;
        init_x(x, nx, ny, nz);

        if (argc != 5) {
                printf("usage: %s <filename> <nx> <ny> <nz>\n", argv[0]);
                return -1;
        }

        
        printf("Writing %dx%dx%d volume to %s \n", nx, ny, nz, filename);
        write_volume(filename, x, nx, ny, nz);
        free(x);
        
}
