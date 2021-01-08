#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VERBOSE 0
#define NVCC
#include "printing.cuh"

#include "cuda_err_check.h"
#include "read_volume.h"
#include "write_volume.h"
#include "wavelet_slow.h"
#include "wavelet_slow.cuh"
#include "opt_32_6.cuh"
#include "compare.h"
#include "diff.h"
#include "norms.h"
#include "init_x.h"
#include "init_random.h"

const int FORWARD = 0;
const int INVERSE = 1;
const int CPU_COMPUTE = 0;
const int ERR_CHECK = 1;

const int RUN_32x32x32 = 1;
const int RUN_8x8x8 = 0;


int main(int argc, char **argv) {

        
        const char *filename = argv[1];
        const char *outfilename = argv[2];
        
        int nx, ny, nz, bx, by, bz;
        float *x, *x2;

        if (filename) {
                printf("reading: %s \n", filename);
                read_volume(filename, x, nx, ny, nz, bx, by, bz);
                read_volume(filename, x2, nx, ny, nz, bx, by, bz);
        } else {

                nx = 32;
                ny = 32; 
                nz = 32;
                bx = 1;
                by = 1;
                bz = 1;

                init_random(x, nx, ny, nz, bx, by, bz);


        }
        printf("block dimension: %d %d %d \n", nx, ny, nz);
        printf("number of blocks: %d %d %d \n", bx, by, bz);
        size_t num_bytes = sizeof(float) * nx * ny * nz * bx * by * bz;
        float *work = (float*)malloc(num_bytes);
        float *x_gpu = (float*)malloc(num_bytes);
        float *err = (float*)malloc(num_bytes);

        float *d_x;
        cudaMalloc((void**)&d_x, num_bytes);
        cudaMemcpy(d_x, x2, num_bytes, cudaMemcpyHostToDevice);

        int x0 = 0;
        int y0 = 0;
        int z0 = 0;

        int b = bx * by * bz;
        int n = nx * ny * nz;


        if (CPU_COMPUTE) {

        printf("Computing CPU forward transform (single block) ... \n");
        Wavelet_Transform_Slow_Forward(x, work, 32, 32, 32, x0, y0, z0, 32, 32, 32);

        printf("Computing CPU inverse transform (single block) ... \n");
        Wavelet_Transform_Slow_Inverse(x, work, 32, 32, 32, x0, y0, z0, 32, 32, 32);

        const char *errtype[] = {"abs.", "rel."};
        for (int a = 0; a < 2; ++a) {
        double l2err = l2norm(x, x2, b * n, a);
        double l1err = l1norm(x, x2, b * n, a);
        double linferr = linfnorm(x, x2, b * n, a);
        printf("%s l2 error = %g l1 error = %g linf error = %g \n", errtype[a], l2err, l1err, linferr);
        }
        }

        if (RUN_32x32x32) {

        cudaEvent_t start, stop;
        float elapsed = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        printf("[32, 32, 32] Computing GPU forward transform... \n");
        cudaEventRecord(start);
        opt6wl79_32x32x32_h<FORWARD>(d_x, bx, by, bz);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);

        cudaDeviceSynchronize();
        printf("Throughput: %g Mcells/s \n", b * n / elapsed / 1e3); 

        printf("[32, 32, 32] Computing GPU inverse transform... \n");
        cudaEventRecord(start);
        wl79_32x32x32_h<INVERSE>(d_x, bx, by, bz);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("Throughput: %g Mcells/s \n", b * n / elapsed / 1e3); 

        if (ERR_CHECK) {
                printf("Running error checking... \n");
                cudaMemcpy(x_gpu, d_x, num_bytes, cudaMemcpyDeviceToHost);

                //print_array(x_gpu, 32, 32, 32, 0, 0, 0, 4, 4, 4);
                //print_array(x, 32, 32, 32, 0, 0, 0, 4, 4, 4);
                //assert(compare(x, x_gpu, 8, 8, 8, 1));

                const char *errtype[] = {"abs.", "rel."};
                for (int a = 0; a < 2; ++a) {
                double l2err = l2norm(x2, x_gpu, b * n, a);
                double l1err = l1norm(x2, x_gpu, b * n, a);
                double linferr = linfnorm(x2, x_gpu, b * n, a);
                printf("%s l2 error = %g l1 error = %g linf error = %g \n", errtype[a], l2err, l1err, linferr);
                }
        }
        }


        if (RUN_8x8x8) {

        cudaErrCheck(cudaMemcpy(d_x, x2, num_bytes, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        float elapsed = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        printf("[8, 8, 8] Computing GPU forward transform... \n");
        cudaEventRecord(start);
        wl79_8x8x8_h<FORWARD>(d_x, bx, by, bz);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsed, start, stop);
        cudaDeviceSynchronize();
        printf("Throughput: %g Mcells/s \n", b * n / elapsed / 1e3); 


        printf("[8, 8, 8] Computing GPU inverse transform... \n");
        elapsed = 0;
        cudaEventRecord(start);
        wl79_8x8x8_h<INVERSE>(d_x, bx, by, bz);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsed, start, stop);
        cudaDeviceSynchronize();
        printf("Throughput: %g Mcells/s \n", b * n / elapsed / 1e3); 
        cudaDeviceSynchronize();

        if (ERR_CHECK) {
                printf("Running error checking... \n");
                cudaMemcpy(x_gpu, d_x, num_bytes, cudaMemcpyDeviceToHost);
                assert(compare(x, x_gpu, 8, 8, 8, 1, 1e-3f));

                const char *errtype[] = {"abs.", "rel."};
                for (int a = 0; a < 2; ++a) {
                double l2err = l2norm(x2, x_gpu, b * n, a);
                double l1err = l1norm(x2, x_gpu, b * n, a);
                double linferr = linfnorm(x2, x_gpu, b * n, a);
                printf("%s l2 error = %g l1 error = %g linf error = %g \n", errtype[a], l2err, l1err, linferr);
                }
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
        }
        
        printf("Test(s) passed!\n");


        if (outfilename) {
                printf("writing: %s \n", outfilename);
                write_volume(outfilename, x, nx, ny, nz, bx, by, bz);
        }

        
}

