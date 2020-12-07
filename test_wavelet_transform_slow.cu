#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VERBOSE 0
#define NVCC
#include "printing.cuh"

#include "read_volume.h"
#include "write_volume.h"
#include "wavelet_slow.h"
#include "wavelet_slow.cuh"
#include "compare.h"
#include "diff.h"
#include "norms.h"

const int FORWARD = 0;
const int INVERSE = 1;

int main(int argc, char **argv) {

        
        const char *filename = argv[1];
        const char *outfilename = argv[2];

        int nx, ny, nz, bx, by, bz;

        float *x, *x2;
        printf("reading: %s \n", filename);
        read_volume(filename, x, nx, ny, nz, bx, by, bz);
        read_volume(filename, x2, nx, ny, nz, bx, by, bz);
        printf("block dimension: %d %d %d \n", nx, ny, nz);
        printf("number of blocks: %d %d %d \n", bx, by, bz);
        size_t num_bytes = sizeof(float) * nx * ny * nz * bx * by * bz;
        float *work = (float*)malloc(num_bytes);
        float *x_gpu = (float*)malloc(num_bytes);
        float *err = (float*)malloc(num_bytes);

        float *d_x;
        cudaMalloc((void**)&d_x, num_bytes);
        cudaMemcpy(d_x, x, num_bytes, cudaMemcpyHostToDevice);

        int x0 = 0;
        int y0 = 0;
        int z0 = 0;

        int b = bx * by * bz;
        int n = nx * ny * nz;


        {

        printf("Computing CPU forward transform... \n");
        Wavelet_Transform_Slow_Forward(x, work, 8, 8, 8, x0, y0, z0, 8, 8, 8);
        printf("Computing CPU inverse transform... \n");
        Wavelet_Transform_Slow_Inverse(x, work, 8, 8, 8, x0, y0, z0, 8, 8, 8);

        //assert(compare(x, x2, 8, 8, 8, 1));

        double l2err = l2norm(x, x2, b * n);
        double l1err = l1norm(x, x2, b * n);
        double linferr = linfnorm(x, x2, b * n);
        printf("l2 error = %g l1 error = %g linf error = %g \n", l2err, l1err, linferr);
        }

        {
        dim3 threads(32, 2, 1);
        dim3 blocks(bx, by, bz);
        printf("Computing GPU forward transform... \n");
        wl79_8x8x8<FORWARD><<<blocks, threads>>>(d_x);
        cudaDeviceSynchronize();
        printf("Computing GPU inverse transform... \n");
        wl79_8x8x8<INVERSE><<<blocks, threads>>>(d_x);
        cudaDeviceSynchronize();
        cudaMemcpy(x_gpu, d_x, num_bytes, cudaMemcpyDeviceToHost);
        assert(compare(x, x_gpu, 8, 8, 8, 1));

        const char *errtype[] = {"abs.", "rel."};
        for (int a = 0; a < 2; ++a) {
        double l2err = l2norm(x, x_gpu, b * n, a);
        double l1err = l1norm(x, x_gpu, b * n, a);
        double linferr = linfnorm(x, x_gpu, b * n, a);
        printf("%s l2 error = %g l1 error = %g linf error = %g \n", errtype[a], l2err, l1err, linferr);
        }
        return -1;
        }

        if (VERBOSE) {
                diff(err, x, x_gpu, 512);
                printf("x = \n");
                print_array(x, 8, 8, 8);
                printf("x_gpu = \n");
                print_array(x_gpu, 8, 8, 8);
                printf("err = \n");
                print_array(err, 8, 8, 8);
        }
        
        printf("Test(s) passed!\n");


        if (outfilename) {
                printf("writing: %s \n", outfilename);
                write_volume(outfilename, x, nx, ny, nz, bx, by, bz);
        }
        
}

