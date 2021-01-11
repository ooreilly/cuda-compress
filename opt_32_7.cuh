#pragma once
#include "opt32.cuh"

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

        const int num_batches_z = (32 - 1) / planes + 1;

        register float plane[4][32];

        for (int batch_z = 0; batch_z < num_batches_z; ++batch_z) {
                      // Load all (x,y) planes into shared memory  
                      // Process an entire 32 x 32 plane
              if (batch_z * block_y + idy < 32) {
                      for (int tile_y = 0; tile_y < 32; ++tile_y) {
                        size_t sptr = idx + snx * tile_y + snxy * idy ;
                        size_t gptr = idx + 32 * tile_y + 1024 * idy;
                        smem[sptr] = in[batch_z * planes * 1024 + gptr + block_idx];
                      }
              }

              __syncthreads();

              // Apply wavelet transform line by line in the x-direction
              opt7ds79_compute<kernel>(&smem[snx * idx + snxy * idy], 1);

              __syncthreads();

              // Apply wavelet transform line by line in the y-direction
              opt7ds79_compute<kernel>(
                                  &smem[idx + snxy * idy],
                                  snx);

                __syncthreads();

              // Write result to global memory
              // Write all (x,y) planes back to global memory  
              // Process an entire 32 x 32 plane
                for (int tile_y = 0; tile_y < 32; ++tile_y) {
                        size_t sptr = idx + snx * tile_y + snxy * idy;
                        size_t gptr = idx + 32 * tile_y + 1024 * idy;
                        in[batch_z * planes * 1024 + gptr + block_idx] =
                            smem[sptr];
                }

                __syncthreads();

        }

        
       const int num_batches_y = (32 - 1) / planes + 1;

        for (int batch_y = 0; batch_y < num_batches_y; ++batch_y) { 

              // Load all (x,z) planes into shared memory  
              if (batch_y * block_y + idy < 32) {
                      // Process an entire 32 x 32 plane
                      for (int tile_z = 0; tile_z < 32 ; ++tile_z) {
                        size_t sptr = idx + snx * tile_z  +  snxy * idy;
                        size_t gptr = idx + 1024 * tile_z + 32 * idy;
                        smem[sptr] = in[batch_y * planes * 32 + gptr + block_idx];
                      }
                }

              __syncthreads();

              // Apply wavelet transform line by line in the z-direction
              opt7ds79_compute<kernel>(&smem[idx + snxy * idy], snx);

              __syncthreads();

              // Write all (x,z) planes back to global memory  
              if (batch_y * block_y + idy < 32) {
                      // Process an entire 32 x 32 plane
                      for (int tile_z = 0; tile_z < 32 ; ++tile_z) {
                        size_t sptr = idx + snx * tile_z + snxy * idy;
                        size_t gptr = idx + 1024 * tile_z  + 32 * idy;
                        in[batch_y * planes * 32 + gptr + block_idx] = smem[sptr];
                      }
              }
              
              __syncthreads();
        }

}

template <int mode>
void opt7wl79_32x32x32_h(float *in, const int bx, const int by, const int bz) {
        const int block_y = 8;
        dim3 threads(32, block_y, 1);
        dim3 blocks(bx, by, bz);
        //printf("blocks: %d %d %d \n", blocks.x, blocks.y, blocks.z);
        //const size_t smem = 1 << 15;


        //int carveout = cudaSharedmemCarveoutMaxShared;
        //cudaErrCheck(cudaFuncSetAttribute(
        //    opt4wl79_32x32x32<mode, block_y>, cudaFuncAttributePreferredSharedMemoryCarveout,
        //    carveout));
 
        //cudaErrCheck(cudaDeviceSetCacheConfig (cudaFuncCachePreferL1));

        //cudaErrCheck(cudaFuncSetAttribute(
        //            opt4wl79_32x32x32<mode, block_y>, 
        //            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

        opt7wl79_32x32x32<mode, block_y><<<blocks, threads>>>(in);
        //opt4wl79_32x32x32<mode, block_y><<<blocks, threads>>>(in);
        cudaErrCheck(cudaPeekAtLastError());
}

