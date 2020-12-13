template <int kernel, int block_y>
__launch_bounds__(32 * block_y)
__global__ void opt1wl79_32x32x32(float *in) {

        int idx = threadIdx.x;
        int idy = threadIdx.y;

        const int planes = block_y;
        const int block_size = 32 * 32 * 32;

        const int snx = 33;
        const int sny = 32;
        const int snxy = snx * sny;

        __shared__ float smem[snxy * planes];
        __shared__ float smem2[snxy * planes];

        size_t block_idx =
            block_size *
            (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));

        const int num_batches_z = 32 / planes;

        for (int batch_z = 0; batch_z < num_batches_z; ++batch_z) {
              // Load all (x,y) planes into shared memory  
              for (int z = 0; z < planes; ++z) {
                      // Process an entire 32 x 32 plane
                      for (int tile_y = 0; tile_y < 32 / block_y; ++tile_y) {
                        size_t sptr = idx + snx * (tile_y * block_y + idy) + snxy * z;
                        size_t gptr = idx + 32 * (tile_y * block_y + idy) + 1024 * z;
                        smem[sptr] = in[batch_z * planes * 1024 + gptr + block_idx];
                      }
              }

              __syncthreads();

              // Apply wavelet transform line by line in the x-direction
              for (int z = 0; z < planes / block_y; ++z) {
                      if (kernel == 0) { 
                        ds79_compute_shared<32>(
                            &smem[snx * idx + snxy * (idy + z * block_y)],
                            &smem2[snx * idx + snxy * (idy + z * block_y)], 1);
                      } else {
                      us79_compute_shared<32>(
                          &smem[snx * idx + snxy * (idy + z * block_y)],
                          &smem2[snx * idx + snxy * (idy + z * block_y)], 1);
                        }
               }

                __syncthreads();

              // Apply wavelet transform line by line in the y-direction
              for (int z = 0; z < planes / block_y; ++z) {
                      if (kernel == 0) {
                              ds79_compute_shared<32>(
                                  &smem[idx + snxy * (idy + z * block_y)],
                                  &smem2[idx + snxy * (idy + z * block_y)], snx);
                      } else {
                              us79_compute_shared<32>(
                                  &smem[idx + snxy * (idy + z * block_y)],
                                  &smem2[idx + snxy * (idy + z * block_y)], snx);
                      }
              }

                __syncthreads();

              // Write result to global memory
              // Write all (x,y) planes back to global memory  
              for (int z = 0; z < planes; ++z) {
                      // Process an entire 32 x 32 plane
                      for (int tile_y = 0; tile_y < 32 / block_y; ++tile_y) {
                        size_t sptr = idx + snx * (tile_y * block_y + idy) + snxy * z;
                        size_t gptr = idx + 32 * (tile_y * block_y + idy) + 1024 * z;
                        in[batch_z * planes * 1024 + gptr + block_idx] = smem[sptr];
                      }
              }

        __syncthreads();

        }
        
       const int num_batches_y = 32 / planes;

        for (int batch_y = 0; batch_y < num_batches_y; ++batch_y) { 

              // Load all (x,z) planes into shared memory  
              for (int y = 0; y < planes; ++y) {
                      // Process an entire 32 x 32 plane
                      for (int tile_z = 0; tile_z < 32 / block_y; ++tile_z) {
                        size_t sptr = idx + snx * (tile_z * block_y + idy) + snxy * y;
                        size_t gptr = idx + 1024 * (tile_z * block_y + idy) + 32 * y;
                        smem[sptr] = in[batch_y * planes * 32 + gptr + block_idx];
                      }
              }

              __syncthreads();

              // Apply wavelet transform line by line in the z-direction
              for (int y = 0; y < planes / block_y; ++y) {
                      if (kernel == 0) {
                        ds79_compute_shared<32>(
                            &smem[idx + snxy * (idy + y * block_y)],
                            &smem2[idx + snxy * (idy + y * block_y)], snx);
                      } else {
                        us79_compute_shared<32>(
                            &smem[idx + snxy * (idy + y * block_y)],
                            &smem2[idx + snxy * (idy + y * block_y)], snx);
                      }
               }


              __syncthreads();

              // Write all (x,z) planes back to global memory  
              for (int y = 0; y < planes; ++y) {
                      // Process an entire 32 x 32 plane
                      for (int tile_z = 0; tile_z < 32 / block_y; ++tile_z) {
                        size_t sptr = idx + snx * (tile_z * block_y + idy) + snxy * y;
                        size_t gptr = idx + 1024 * (tile_z * block_y + idy) + 32 * y;
                        in[batch_y * planes * 32 + gptr + block_idx] = smem[sptr];
                      }
              }
        }

}

template <int mode>
void opt1wl79_32x32x32_h(float *in, const int bx, const int by, const int bz) {
        const int block_y = 4;
        dim3 threads(32, block_y, 1);
        dim3 blocks(bx, by, bz);
        opt1wl79_32x32x32<mode, block_y><<<blocks, threads>>>(in);
        cudaErrCheck(cudaPeekAtLastError());
}

