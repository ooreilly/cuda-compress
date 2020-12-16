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
#include "compare.h"
#include "diff.h"
#include "norms.h"
#include "init_x.h"
#include "init_random.h"

const int FORWARD = 0;
const int INVERSE = 1;


int err_check(float *x_gpu, float *d_x, float *x, const int nx, const int ny, const int nz, 
                const int bx, const int by, const int bz, const bool verbose=false, 
                const double l2_tol=1e-5, const double l1_tol=1e-5, const double linf_tol=1e-5) {
        size_t b = bx * by * bz;
        size_t n = nx * ny * nz;
        size_t num_bytes = b * n * sizeof(float);
        cudaMemcpy(x_gpu, d_x, num_bytes, cudaMemcpyDeviceToHost);


        const char *errtype[] = {"abs.", "rel."};
        for (int a = 0; a < 2; ++a) {
                double l2err = l2norm(x, x_gpu, b * n, a);
                double l1err = l1norm(x, x_gpu, b * n, a);
                double linferr = linfnorm(x, x_gpu, b * n, a);
                if (verbose) printf("%s l2 error = %g l1 error = %g linf error = %g \n",
                       errtype[a], l2err, l1err, linferr);
                if (a == 1 && (l2err > l2_tol || l1err > l1_tol || linferr > linf_tol) ) return 1;
        }
        return 0;
}

void print_status(int err) {
        if (!err) printf("OK\n");
        else printf("FAILED\n");
}


int test_kernel(enum kernel k, const int nx, const int ny, const int nz, const int bx, const int by, const int bz, const int verbose) {

        float *x;
        init_random(x, nx, ny, nz, bx, by, bz);

        size_t num_bytes = sizeof(float) * nx * ny * nz * bx * by * bz;
        float *x_gpu = (float*)malloc(num_bytes);

        float *d_x;
        cudaMalloc((void**)&d_x, num_bytes);
        cudaMemcpy(d_x, x, num_bytes, cudaMemcpyHostToDevice);

        printf("%s \t [%d, %d, %d] [%d, %d, %d] \n", get_kernel_name(k), nx, ny, nz, bx, by, bz);
        wl79_h<FORWARD>(k, d_x, bx, by, bz);
        wl79_h<INVERSE>(k, d_x, bx, by, bz);
        cudaDeviceSynchronize();
        int err = err_check(x_gpu, d_x, x, nx, ny, nz, bx, by, bz, verbose);
        print_status(err);
        free(x);
        free(x_gpu);
        cudaFree(d_x);

        return err;
}


int main(int argc, char **argv) {

        
        const int verbose = 1;

        int bx = 11;
        int by = 9;
        int bz = 8;
        test_kernel(WL79_8x8x8, bz, bz, bz, bx, by, bz, verbose);
        test_kernel(WL79_32x32x32, 32, 32, 32, bx, by, bz, verbose);
        test_kernel(OPT1WL79_32x32x32, 32, 32, 32, bx, by, bz, verbose);
        test_kernel(OPT2WL79_32x32x32, 32, 32, 32, bx, by, bz, verbose);
        test_kernel(OPT3WL79_32x32x32, 32, 32, 32, bx, by, bz, verbose);
        test_kernel(OPT4WL79_32x32x32, 32, 32, 32, bx, by, bz, verbose);
}

