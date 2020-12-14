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

template <int mode>
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

        const char* modes[] = {"Forward", "Inverse"};

        cudaEventRecord(start);
        wl79_h<mode>(k, d_x, bx, by, bz);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaDeviceSynchronize();
        printf(
            "%-20s \t %s \t %7s [%d, %d, %d] \t %7s [%d, %d, %d] \t %g Mcells/s\n",
            get_kernel_name(k), modes[mode], "", nx, ny, nz, "", bx * nx, by * ny, bz * nz,
            b * n / elapsed / 1e3);
        free(x);
        cudaFree(d_x);
}

int main(int argc, char **argv) {

        printf("Kernel name       \t Wavelet transform \t Block dimension \t Grid dimension \t Throughput\n");

        //transform<FORWARD>(WL79_8x8x8, 8, 8, 8, 44, 52, 40);
        //transform<FORWARD>(WL79_8x8x8, 8, 8, 8, 88, 104, 80);
        //transform<FORWARD>(WL79_8x8x8, 8, 8, 8, 132, 156, 120);

        transform<FORWARD>(WL79_32x32x32, 32, 32, 32, 10, 12, 13);
        //transform<FORWARD>(WL79_32x32x32, 32, 32, 32, 20, 25, 20);
        //transform<FORWARD>(WL79_32x32x32, 32, 32, 32, 40, 32, 18);

        transform<FORWARD>(OPT1WL79_32x32x32, 32, 32, 32, 10, 12, 13);
        transform<FORWARD>(OPT2WL79_32x32x32, 32, 32, 32, 10, 12, 13);
        //transform<FORWARD>(OPT1WL79_32x32x32, 32, 32, 32, 20, 25, 20);
        //transform<FORWARD>(OPT1WL79_32x32x32, 32, 32, 32, 40, 32, 18);
        //
        transform<INVERSE>(OPT1WL79_32x32x32, 32, 32, 32, 10, 12, 13);
        transform<INVERSE>(OPT2WL79_32x32x32, 32, 32, 32, 10, 12, 13);
        
}

