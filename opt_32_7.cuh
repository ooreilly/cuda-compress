#pragma once
#include "opt32.cuh"
#include "opt_32_7_extra.cuh"
#include "printing.cuh"


template <int kernel>
inline __device__ void opt7ds79_compute(float *p_in, int stride) {
        if (kernel == 0) {
                opt5ds79_compute2(p_in, stride);
        } else {
                us79_compute<32>(p_in, stride);
        }
}

template <int kernel, int block_y>
__launch_bounds__(32 * block_y, 1)
__global__ void opt7wl79_32x32x32(float *in) {

        int idx = threadIdx.x;
        int idy = threadIdx.y;

        const int planes = block_y;
        const int block_size = 32 * 32 * 32;

        const int snx = 33;
        const int sny = 32;
        const int snxy = snx * sny;

        __shared__ float smem[snxy * planes];

        size_t block_idx =
            block_size *
            (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));

        const int num_batches_z = 4; 


        // Each thread holds 4 x 32 registers, denoted by p_z_y that correspond to four y-lines of
        // strided data in the z-direction. The stride depends on the number of warps. 
        // If the cube is A[z, y, x] (x fast),
        // then thread 0 holds A[0, y, 0], A[8, y, 0], A[16, y, 0], A[24, y, 0].
        //register float p[num_batches_z][32];
        DECLARE_REGISTERS

        for (int batch_z = 0; batch_z < num_batches_z; ++batch_z) {
              // Load all (x,y) planes into shared memory  
              // Process an entire 32 x 32 plane
              for (int tile_y = 0; tile_y < 32; ++tile_y) {
                        size_t sptr = idx + snx * tile_y + snxy * idy ;
                        size_t gptr = idx + 32 * tile_y + 1024 * idy;
                        smem[sptr] = in[batch_z * planes * 1024 + gptr + block_idx];
              }

              __syncwarp();

              // Apply wavelet transform line by line in the x-direction
              opt7ds79_compute<kernel>(&smem[snx * idx + snxy * idy], 1);

              __syncwarp();

              // Apply wavelet transform line by line in the y-direction
              opt7ds79_compute<kernel>(
                                  &smem[idx + snxy * idy],
                                  snx);

                __syncwarp();

              // Write result to global memory
              // Write all (x,y) planes back to global memory  
              // Process an entire 32 x 32 plane
                //for (int tile_y = 0; tile_y < 32; ++tile_y) {
                //        size_t sptr = idx + snx * tile_y + snxy * idy;
                //        size_t gptr = idx + 32 * tile_y + 1024 * idy;
                //        in[batch_z * planes * 1024 + gptr + block_idx] =
                //            smem[sptr];
                //}

                // Load shared memory data into registers
                // for (int j = 0; j < 32; ++j)
                //        p[batch_z][j] = smem[idx + snx * j + snxy * idy];
                LOAD_SHARED(batch_z)

                __syncthreads();

        }
        
       const int num_batches_y = 4;

        for (int batch_y = 0; batch_y < num_batches_y; ++batch_y) { 

              // Load (x,z) planes from registers and store in shared memory
              //for (int y = 0; y < 8; ++y) {
              //for (int plane = 0; plane < 4; ++plane) {
              //        int sptr = idx + snx * 8 * plane + snx * idy + snxy * y;
              //        smem[sptr] = p[plane][y + 8 * batch_y];
              //}
              //}
              STORE_SHARED(batch_y)

              __syncthreads();

              // Apply wavelet transform line by line in the z-direction
              opt7ds79_compute<kernel>(&smem[idx + snxy * idy], snx);

              __syncwarp();

              // Write all (x,z) planes back to global memory
              for (int tile_z = 0; tile_z < 32; ++tile_z) {
                      size_t sptr = idx + snx * tile_z + snxy * idy;
                      size_t gptr = idx + 1024 * tile_z + 32 * idy;
                      in[batch_y * planes * 32 + gptr + block_idx] = smem[sptr];
              }

              __syncthreads();
        }

}

template <int mode>
void opt7wl79_32x32x32_h(float *in, const int bx, const int by, const int bz) {
        const int block_y = 8;
        dim3 threads(32, block_y, 1);
        dim3 blocks(bx, by, bz);
        opt7wl79_32x32x32<mode, block_y><<<blocks, threads>>>(in);
        cudaErrCheck(cudaPeekAtLastError());
}

