#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "printing.cuh"

#include "read_volume.h"
#include "write_volume.h"
#include "wavelet_slow.h"

int main(int argc, char **argv) {

        
        const char *filename = argv[1];
        const char *outfilename = argv[2];

        float *x;
        printf("reading: %s \n", filename);
        int nx = 0, ny = 0, nz = 0;
        read_volume(filename, x, nx, ny, nz);
        size_t num_bytes = sizeof(float) * nx * ny * nz * 1024;
        float *work = (float*)malloc(num_bytes);

        printf("transforming... \n");
        int bx = 8;
        int by = 8;
        int bz = 8;
        int x0 = 0;
        int y0 = 0;
        int z0 = 0;

        //for (int iy = 0;  iy < by;  ++iy) 
                Ds79(&x[nx*0], work,  1, bx);
                print_array(x, nx, 1, 1);
        //for (int ix = 0;  ix < bx;  ++ix) 
        //        Ds79(&x[ix], work, bx, by);


        printf("writing: %s \n", outfilename);
        write_volume(outfilename, x, nx, ny, 1);

        free(x);
        free(work);
        
}

