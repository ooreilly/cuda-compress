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
#include "init_random.h"
#include "cuda_err_check.h"

const int FORWARD = 0;
const int INVERSE = 1;

void transform(enum kernel k, int nx, int ny, int nz, int bx, int by, int bz) {

        size_t b = bx * by * bz;
        size_t n = nx * ny * nz;

        size_t num_bytes = sizeof(float) * nx * ny * nz * bx * by * bz;

        float *x;
        init_random(x, nx, ny, nz, bx, by, bz);

        float *d_x;
        cudaErrCheck(cudaMalloc((void**)&d_x, num_bytes));
        cudaErrCheck(cudaMemcpy(d_x, x, num_bytes, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        float elapsed = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        wl79_h<FORWARD>(k, d_x, bx, by, bz);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaDeviceSynchronize();
        printf(
            "Forward \t %7s [%d, %d, %d] \t %7s [%d, %d, %d] \t %g Mcells/s\n",
            "", nx, ny, nz, "", bx * nx, by * ny, bz * nz,
            b * n / elapsed / 1e3);
        cudaEventRecord(start);
        wl79_h<INVERSE>(k, d_x, bx, by, bz);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaDeviceSynchronize();
        printf(
            "Inverse \t %7s [%d, %d, %d] \t %7s [%d, %d, %d] \t %g Mcells/s\n",
            "", nx, ny, nz, "", bx * nx, by * ny, bz * nz,
            b * n / elapsed / 1e3);

        free(x);
        cudaFree(d_x);
}

int main(int argc, char **argv) {

        printf("Wavelet transform \t Block dimension \t Grid dimension \t Throughput\n");

        transform(WL79_8x8x8, 8, 8, 8, 44, 52, 40);
        transform(WL79_8x8x8, 8, 8, 8, 88, 104, 80);
        transform(WL79_8x8x8, 8, 8, 8, 132, 156, 120);

        transform(WL79_32x32x32, 32, 32, 32, 10, 12, 13);
        transform(WL79_32x32x32, 32, 32, 32, 20, 25, 20);
        transform(WL79_32x32x32, 32, 32, 32, 40, 32, 18);
        
}

