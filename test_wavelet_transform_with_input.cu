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
#include "init_x.h"

const int FORWARD = 0;
const int INVERSE = 1;


void transform(int nx, int ny, int nz, int bx, int by, int bz) {

        printf("block dimension: %d %d %d \n", nx, ny, nz);
        printf("number of blocks: %d %d %d \n", bx, by, bz);

        size_t b = bx * by * bz;
        size_t n = nx * ny * nz;

        size_t num_bytes = sizeof(float) * nx * ny * nz * bx * by * bz;

        float *x;
        init_x(x, nx, ny, nz, bx, by, bz);

        float *d_x;
        cudaMalloc((void**)&d_x, num_bytes);
        cudaMemcpy(d_x, x, num_bytes, cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        float elapsed = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        dim3 threads(32, 2, 1);
        dim3 blocks(bx, by, bz);

        printf("Computing GPU forward transform... \n");
        {
                cudaEventRecord(start);
                wl79_8x8x8<FORWARD><<<blocks, threads>>>(d_x);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsed, start, stop);
                cudaDeviceSynchronize();
                printf("Throughput: %g Mcells/s \n", b * n / elapsed / 1e3); 
        }


        printf("Computing GPU inverse transform... \n");
        {
                cudaEventRecord(start);
                wl79_8x8x8<INVERSE><<<blocks, threads>>>(d_x);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsed, start, stop);
                cudaDeviceSynchronize();
                printf("Throughput: %g Mcells/s \n", b * n / elapsed / 1e3); 
        }

        printf("\n");

        free(x);
        cudaFree(d_x);
}

int main(int argc, char **argv) {

        transform(8, 8, 8, 44, 52, 40);
        transform(8, 8, 8, 88, 104, 80);
        transform(8, 8, 8, 132, 156, 120);
        
}

