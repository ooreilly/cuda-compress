#include <stdio.h>
#include <stdlib.h>
#include "write_volume.h"

// Write bx * by * bz blocks each of dimension nx * ny * nz
void init_x(float*& out, int nx, int ny, int nz, int bx, int by, int bz) {

        size_t block = bx * by * bz;
        out = (float*)malloc(sizeof(float) * block * nx * ny * nz);

        // Number of blocks
        for (int jz = 0; jz < bz; ++jz)
        for (int jy = 0; jy < by; ++jy)
        for (int jx = 0; jx < bx; ++jx) {
                // Block dimension
                for (int iz = 0; iz < nz; ++iz)
                for (int iy = 0; iy < ny; ++iy)
                for (int ix = 0; ix < nx; ++ix) {
                        size_t pos_block =  ix + nx * (iy + ny * iz);
                        size_t pos = pos_block + block * (jx + bx * (jy + by * jz));
                        out[pos] = ix + nx * jx;// * iy * iz;      
                }
        }
}


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
