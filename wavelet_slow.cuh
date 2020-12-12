#pragma once
#include <stdio.h>
#include <cuda.h>
#include "cuda_err_check.h"

#define al0  8.526986790094000e-001f
#define al1  3.774028556126500e-001f
#define al2 -1.106244044184200e-001f
#define al3 -2.384946501938001e-002f
#define al4  3.782845550699501e-002f

#define ah0  7.884856164056601e-001f
#define ah1 -4.180922732222101e-001f
#define ah2 -4.068941760955800e-002f
#define ah3  6.453888262893799e-002f

inline __device__ int dMIRR(int inp_val, int dim)
{
	int val = inp_val < 0 ? -inp_val : inp_val;
	val = (val >= dim) ? (2*dim-2-val) : val;
	val = val < 0 ? -val : val;
	val = (val >= dim) ? (2*dim-2-val) : val;
	return val;
}

inline __device__ int dMIRR_SL(int inp_val, int nl)
{
	int val = inp_val;
	val = val < 0 ? -val : val;
	val = (val >= nl) ? (2*nl-1-val) : val;
	val = val < 0 ? -val : val;
	val = (val >= nl) ? (2*nl-1-val) : val;
	val = val < 0 ? -val : val;
	val = (val >= nl) ? (2*nl-1-val) : val;
	return val;
}


inline __device__ int dMIRR_SH(int inp_val, int nl, int nh)
{          
	int val = inp_val - nl;
	val = val < 0 ? -val-1 : val;
	val = (val >= nh) ? (2*nh-2-val) : val;
	val = val < 0 ? -val-1 : val;
	val = (val >= nh) ? (2*nh-2-val) : val;
	val = val < 0 ? -val-1 : val;
	val = (val >= nh) ? (2*nh-2-val) : val;
	return nl + val;
}

template <int size>
inline __device__ void ds79_compute(float *p_in, int stride) {
        float p_tmp[size];

	for (int n = size;  n >= 2;  n = n-n/2)
	{

		// copy inputs to tmp buffer, p_in will be overwritten
		for (int i = 0;  i < n;  ++i) p_tmp[i] = p_in[i*stride];

		int nh = n / 2;
		int nl = n - nh;
		for (int ix = 0;  ix < nl;  ++ix)
		{

			int i0 = 2 * ix;
			int im1 = dMIRR(i0-1,n);  int ip1 = dMIRR(i0+1,n);
			int im2 = dMIRR(i0-2,n);  int ip2 = dMIRR(i0+2,n);
			int im3 = dMIRR(i0-3,n);  int ip3 = dMIRR(i0+3,n);
			int im4 = dMIRR(i0-4,n);  int ip4 = dMIRR(i0+4,n);

			// sum smallest to largest (most accurate way of summing floats)
			float acc1 = al4 * (p_tmp[im4] + p_tmp[ip4]);
			acc1 += al1 * (p_tmp[im1] + p_tmp[ip1]);
			acc1 += al0 * p_tmp[i0];
			float acc2 = al3 * (p_tmp[im3] + p_tmp[ip3]);
			acc2 += al2 * (p_tmp[im2] + p_tmp[ip2]);
			p_in[ix*stride] = acc1 + acc2;
		}
		for (int ix = 0;  ix < nh;  ++ix)
		{
			int i0 = 2 * ix + 1;
			int im1 = dMIRR(i0-1,n);  int ip1 = dMIRR(i0+1,n);
			int im2 = dMIRR(i0-2,n);  int ip2 = dMIRR(i0+2,n);
			int im3 = dMIRR(i0-3,n);  int ip3 = dMIRR(i0+3,n);

			float acc1 = ah3 * (p_tmp[im3] + p_tmp[ip3]);
			acc1 += ah0 * p_tmp[i0];
			float acc2 = ah2 * (p_tmp[im2] + p_tmp[ip2]);
			acc2 += ah1 * (p_tmp[im1] + p_tmp[ip1]);
			p_in[(nl+ix)*stride] = acc1 + acc2;
		}
	}

}

template <int size>
inline __device__ void ds79_compute_shared(float *p_in, float *p_tmp, int stride) {

	for (int n = size;  n >= 2;  n = n-n/2)
	{

		// copy inputs to tmp buffer, p_in will be overwritten
		for (int i = 0;  i < n;  ++i) p_tmp[i*stride] = p_in[i*stride];

		int nh = n / 2;
		int nl = n - nh;
		for (int ix = 0;  ix < nl;  ++ix)
		{

			int i0 = 2 * ix;
			int im1 = stride * dMIRR(i0-1,n);  int ip1 = stride * dMIRR(i0+1,n);
			int im2 = stride * dMIRR(i0-2,n);  int ip2 = stride * dMIRR(i0+2,n);
			int im3 = stride * dMIRR(i0-3,n);  int ip3 = stride * dMIRR(i0+3,n);
			int im4 = stride * dMIRR(i0-4,n);  int ip4 = stride * dMIRR(i0+4,n);

			// sum smallest to largest (most accurate way of summing floats)
			float acc1 = al4 * (p_tmp[im4] + p_tmp[ip4]);
			acc1 += al1 * (p_tmp[im1] + p_tmp[ip1]);
			acc1 += al0 * p_tmp[i0 * stride];
			float acc2 = al3 * (p_tmp[im3] + p_tmp[ip3]);
			acc2 += al2 * (p_tmp[im2] + p_tmp[ip2]);
			p_in[ix*stride] = acc1 + acc2;
		}
		for (int ix = 0;  ix < nh;  ++ix)
		{
			int i0 = 2 * ix + 1;
			int im1 = stride * dMIRR(i0-1,n);  int ip1 = stride * dMIRR(i0+1,n);
			int im2 = stride * dMIRR(i0-2,n);  int ip2 = stride * dMIRR(i0+2,n);
			int im3 = stride * dMIRR(i0-3,n);  int ip3 = stride * dMIRR(i0+3,n);

			float acc1 = ah3 * (p_tmp[im3] + p_tmp[ip3]);
			acc1 += ah0 * p_tmp[i0 * stride];
			float acc2 = ah2 * (p_tmp[im2] + p_tmp[ip2]);
			acc2 += ah1 * (p_tmp[im1] + p_tmp[ip1]);
			p_in[(nl+ix)*stride] = acc1 + acc2;
		}
	}

}

inline __device__ void us79_8x8x8_compute(float *p_in, int stride) {
        const int dim = 8;
        float l[8];
        float t[8];
	int nx = 0;
        #pragma unroll
	for (int n = dim;  n >= 2;  n = n-n/2) {l[nx++] = n;}
	for (int li = nx-1;  li >= 0;  --li)
	{
		int n = l[li];

		// copy inputs to tmp buffer, p_in will be overwritten
		for (int i = 0;  i < n;  ++i) t[i] = p_in[i*stride];

		int nh = n / 2;
		int nl = n - nh;
		for (int k = 0;  k < nl;  ++k)
		{
			p_in[2*k*stride] = 
				sl0 * t[k] + 
				sl2 * ( t[dMIRR_SL(k-1,nl)] + t[dMIRR_SL(k+1,nl)] ) +
				sh1 * ( t[dMIRR_SH(nl+k-1,nl,nh)] + t[dMIRR_SH(nl+k,nl,nh)] ) +
				sh3 * ( t[dMIRR_SH(nl+k-2,nl,nh)] + t[dMIRR_SH(nl+k+1,nl,nh)] );
		}
		for (int k = 0;  k < nh;  ++k)
		{
			p_in[(2*k+1)*stride] = 
				sl1 * ( t[dMIRR_SL(k,nl)] + t[dMIRR_SL(k+1,nl)] ) +
				sl3 * ( t[dMIRR_SL(k-1,nl)] + t[dMIRR_SL(k+2,nl)] ) +
				sh0 * t[nl+k] +
				sh2 * ( t[dMIRR_SH(nl+k-1,nl,nh)] + t[dMIRR_SH(nl+k+1,nl,nh)] ) +
				sh4 * ( t[dMIRR_SH(nl+k-2,nl,nh)] + t[dMIRR_SH(nl+k+2,nl,nh)] );
		}
	}
}

template <int kernel>
__global__ void wl79_8x8x8(float *in) {

        int idx = threadIdx.x;
        int idy = threadIdx.y;


        __shared__ float smem[512];

        const int warp_size = 32;

        size_t block_idx = 512 * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));

        for (int z = 0; z < 8; ++z) {
                // Load 2D plane
                // one warp loads 32 B, which gets split over 4 rows
                size_t sptr = idx + warp_size * idy + 64 * z;
                smem[sptr] = in[sptr + block_idx];
        }


        //__syncthreads();
        //if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 1) {
        //        printf("warp 1 grid = %d %d %d \n", gridDim.x, gridDim.y, gridDim.z);
        //        print_array(smem, 8, 8, 8, 0, 0, 0, 4, 4, 8);
        //        //for (int i = 0; i < 512; ++i) printf("%2.2f ", smem[i]);
        //        //printf("\n");
        //}
        //return;

        __syncthreads();
        // Regroup threads for computation (8 x 4)
        int cx = threadIdx.x % 8;
        int cy = threadIdx.x / 8 + 4 * threadIdx.y;

        // Forward
        if (kernel == 0) {
                // Apply wavelet transform line by line in the x-direction
                ds79_compute<8>(&smem[8 * cx + 64 * cy], 1);
                __syncthreads();
                // Apply wavelet transform line by line in the y-direction
                ds79_compute<8>(&smem[cx + 64 * cy], 8);
                __syncthreads();
                // Apply wavelet transform line by line in the z-direction
                ds79_compute<8>(&smem[cy + 8 * cx], 64);
                __syncthreads();
        // Inverse transform
        } else {
                // Apply wavelet transform line by line in the x-direction
                us79_8x8x8_compute(&smem[8 * cx + 64 * cy], 1);
                __syncthreads();
                // Apply wavelet transform line by line in the y-direction
                us79_8x8x8_compute(&smem[cx + 64 * cy], 8);
                __syncthreads();
                // Apply wavelet transform line by line in the z-direction
                us79_8x8x8_compute(&smem[cy + 8 * cx], 64);
                __syncthreads();
        }

        // Write result back
        for (int z = 0; z < 8; ++z) {
                // Load 2D plane
                size_t sptr = idx + warp_size * idy + 64 * z;
                in[sptr + block_idx] = smem[sptr];
        }

        //__syncthreads();
        //if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 1) {
        //        printf("warp 1 grid = %d %d %d \n", gridDim.x, gridDim.y, gridDim.z);
        //        print_array(&in[block_idx], 8, 8, 8, 0, 0, 0, 4, 4, 8);
        //        //for (int i = 0; i < 512; ++i) printf("%2.2f ", smem[i]);
        //        //printf("\n");
        //}
}


template <int mode>
void wl79_8x8x8_h(float *in, const int bx, const int by, const int bz) {
        dim3 threads(32, 2, 1);
        dim3 blocks(bx, by, bz);
        wl79_8x8x8<mode><<<blocks, threads>>>(in);
        cudaErrCheck(cudaPeekAtLastError());
}


template <int kernel, int block_y>
__launch_bounds__(32 * block_y)
__global__ void wl79_32x32x32(float *in) {

        int idx = threadIdx.x;
        int idy = threadIdx.y;

        const int planes = 4;//block_y;
        const int block_size = 32 * 32 * 32;

        __shared__ float smem[1024 * planes];
        __shared__ float smem2[1024 * planes];
        if (threadIdx.x == 0 && threadIdx.y == 0) {
                for (int i = 0; i < 1024 * planes; ++i) smem[i] = 0.0;
        }
        __syncthreads();

        const int warp_size = 32;

        size_t block_idx = block_size * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));

        const int num_batches = 32 / planes;

        for (int batch_z = 0; batch_z < num_batches; ++batch_z) {
              // Load all (x,y) planes into shared memory  
              for (int z = 0; z < planes; ++z) {
                      // Process an entire 32 x 32 plane
                      for (int tile_y = 0; tile_y < 32 / block_y; ++tile_y) {
                        size_t sptr = idx + warp_size * (tile_y * block_y + idy) + 1024 * z;
                        smem[sptr] = in[batch_z * planes * 1024 + sptr + block_idx];
                      }
              }

              __syncthreads();

        //if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
        //        printf("step = %d, warp 1 grid = %d %d %d \n", batch_z, gridDim.x, gridDim.y, gridDim.z);
        //        print_array(smem, 32, 32, planes, 0, 0, 0, 4, 4, planes);
        //        //for (int i = 0; i < 512; ++i) printf("%2.2f ", smem[i]);
        //        //printf("\n");
        //}
        //__syncthreads();
        //
        //


              // Apply wavelet transform line by line in the x-direction
              for (int z = 0; z < planes / block_y; ++z) {
                      ds79_compute_shared<32>(
                          &smem[32 * idx + 1024 * (idy + z * block_y)],
                          &smem2[32 * idx + 1024 * (idy + z * block_y)], 1);
               }

                __syncthreads();
              // Apply wavelet transform line by line in the y-direction
              for (int z = 0; z < planes / block_y; ++z) {
                      ds79_compute_shared<32>(
                          &smem[idx + 1024 * (idy + z * block_y)],
                          &smem2[idx + 1024 * (idy + z * block_y)], 32);
              }

                __syncthreads();

              // Write result to global memory
              //
              // Write all (x,y) planes into shared memory  
              for (int z = 0; z < planes; ++z) {
                      // Process an entire 32 x 32 plane
                      for (int tile_y = 0; tile_y < 32 / block_y; ++tile_y) {
                        size_t sptr = idx + warp_size * (tile_y * block_y + idy) + 1024 * z;
                        in[batch_z * planes * 1024 + sptr + block_idx] = smem[sptr];
                      }
              }

        

        __syncthreads();


        // Load all (x,z) planes into shared memory


        }
        //return;

        // Forward
        //if (kernel == 0) {
        //        // Apply wavelet transform line by line in the x-direction
        //        ds79_8x8x8_compute(&smem[8 * cx + 64 * cy], 1);
        //        __syncthreads();
        //        // Apply wavelet transform line by line in the y-direction
        //        ds79_8x8x8_compute(&smem[cx + 64 * cy], 8);
        //        __syncthreads();
        //        // Apply wavelet transform line by line in the z-direction
        //        ds79_8x8x8_compute(&smem[cy + 8 * cx], 64);
        //        __syncthreads();
        //// Inverse transform
        //} else {
        //        // Apply wavelet transform line by line in the x-direction
        //        us79_8x8x8_compute(&smem[8 * cx + 64 * cy], 1);
        //        __syncthreads();
        //        // Apply wavelet transform line by line in the y-direction
        //        us79_8x8x8_compute(&smem[cx + 64 * cy], 8);
        //        __syncthreads();
        //        // Apply wavelet transform line by line in the z-direction
        //        us79_8x8x8_compute(&smem[cy + 8 * cx], 64);
        //        __syncthreads();
        //}

        // Write result back
        //for (int z = 0; z < 8; ++z) {
        //        // Load 2D plane
        //        size_t sptr = idx + warp_size * idy + 64 * z;
        //        in[sptr + block_idx] = smem[sptr];
        //}

        //__syncthreads();
        //if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 1) {
        //        printf("warp 1 grid = %d %d %d \n", gridDim.x, gridDim.y, gridDim.z);
        //        print_array(&in[block_idx], 8, 8, 8, 0, 0, 0, 4, 4, 8);
        //        //for (int i = 0; i < 512; ++i) printf("%2.2f ", smem[i]);
        //        //printf("\n");
        //}
}

template <int mode>
void wl79_32x32x32_h(float *in, const int bx, const int by, const int bz) {
        const int block_y = 4;
        dim3 threads(32, block_y, 1);
        dim3 blocks(bx, by, bz);
        wl79_32x32x32<mode, block_y><<<blocks, threads>>>(in);
        cudaErrCheck(cudaPeekAtLastError());
}
