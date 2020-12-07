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

        float *x, *x2;
        printf("reading: %s \n", filename);
        int nx = 0, ny = 0, nz = 0;
        read_volume(filename, x, nx, ny, nz);
        read_volume(filename, x2, nx, ny, nz);
        size_t num_bytes = sizeof(float) * nx * ny * nz;
        float *work = (float*)malloc(num_bytes);
        float *x_gpu = (float*)malloc(num_bytes);
        float *err = (float*)malloc(num_bytes);

        float *d_x, *d_y;
        cudaMalloc((void**)&d_x, num_bytes);
        cudaMemcpy(d_x, x, num_bytes, cudaMemcpyHostToDevice);

        int bx = 8;
        int by = 8;
        int bz = 8;
        int x0 = 0;
        int y0 = 0;
        int z0 = 0;


        {
        printf("Computing CPU forward transform... \n");
        Wavelet_Transform_Slow_Forward(x, work, bx, by, bz, x0, y0, z0, nx, ny, nz);
        printf("Computing CPU inverse transform... \n");
        Wavelet_Transform_Slow_Inverse(x, work, bx, by, bz, x0, y0, z0, nx, ny, nz);

        assert(compare(x, x2, 8, 8, 8, 1));

        double l2err = l2norm(x, x2, bx * by * bz);
        double l1err = l1norm(x, x2, bx * by * bz);
        double linferr = linfnorm(x, x2, bx * by * bz);
        printf("l2 error = %g l1 error = %g linf error = %g \n", l2err, l1err, linferr);
        }

        {
        dim3 threads(32, 2, 1);
        dim3 blocks(1, 1, 1);
        printf("Computing GPU forward transform... \n");
        wl79_8x8x8<FORWARD><<<blocks, threads>>>(d_x, bx, by, bz);
        printf("Computing GPU inverse transform... \n");
        wl79_8x8x8<INVERSE><<<blocks, threads>>>(d_x, bx, by, bz);
        cudaDeviceSynchronize();
        cudaMemcpy(x_gpu, d_x, num_bytes, cudaMemcpyDeviceToHost);
        assert(compare(x, x_gpu, 8, 8, 8, 1));

        double l2err = l2norm(x, x_gpu, bx * by * bz);
        double l1err = l1norm(x, x_gpu, bx * by * bz);
        double linferr = linfnorm(x, x_gpu, bx * by * bz);
        printf("l2 error = %g l1 error = %g linf error = %g \n", l2err, l1err, linferr);
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
                write_volume(outfilename, x, nx, ny, 1);
        }
        
}

