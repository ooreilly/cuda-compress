#include "opt32.cuh"

template <int size>
inline __device__ void ds79_compute2(float *p_in, int stride) {
        register float p_tmp[size];

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

                        printf("p: %d %d %d %d %d %d %d %d %d \n",
                                        im4, im3, im2, im1, i0, ip1, ip2, ip3, ip4);

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

                        printf("q: %d %d %d %d %d %d %d \n",
                                        im3, im2, im1, i0, ip1, ip2, ip3);
		}

                printf("n = %d\n", n);
                print_array(p_in, 32, 1, 1);
	}

}
template <int size>
inline __device__ void opt5ds79_compute(float *p_in, int stride) {
        float p_tmp[size];
        opt5ds79_compute2(p_in, stride);

}

template <int kernel, int block_y>
__launch_bounds__(32 * block_y, 1)
__global__ void opt5wl79_32x32x32(float *in) {

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
              if (batch_z * block_y + idy < 32) {
              if (kernel == 0) { 
                opt5ds79_compute<32>(
                    &smem[snx * idx + snxy * idy], 
                    1);
              } else {
              us79_compute<32>(
                  &smem[snx * idx + snxy * idy], 1);
                }
               }

                __syncthreads();

              // Apply wavelet transform line by line in the y-direction
              if (batch_z * block_y + idy < 32) {
                      if (kernel == 0) {
                              opt5ds79_compute<32>(
                                  &smem[idx + snxy * idy],
                                  snx);
                      } else {
                              us79_compute<32>(
                                  &smem[idx + snxy * idy], snx);
                      }
              }

                __syncthreads();

              // Write result to global memory
              // Write all (x,y) planes back to global memory  
              if (batch_z * block_y + idy < 32) {
                      // Process an entire 32 x 32 plane
                      for (int tile_y = 0; tile_y < 32; ++tile_y) {
                        size_t sptr = idx + snx * tile_y + snxy * idy;
                        size_t gptr = idx + 32 * tile_y + 1024 * idy;
                        in[batch_z * planes * 1024 + gptr + block_idx] = smem[sptr];
                      }
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
              if (batch_y * block_y + idy < 32) {
                      if (kernel == 0) {
                        opt5ds79_compute<32>(
                            &smem[idx + snxy * idy],
                            snx);
                      } else {
                        us79_compute<32>(
                            &smem[idx + snxy * idy], snx);
                      }
              }


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
void opt5wl79_32x32x32_h(float *in, const int bx, const int by, const int bz) {
        const int block_y = 1;
        dim3 threads(32, block_y, 1);
        dim3 blocks(bx, by, bz);
        printf("blocks: %d %d %d \n", blocks.x, blocks.y, blocks.z);
        //const size_t smem = 1 << 15;


        //int carveout = cudaSharedmemCarveoutMaxShared;
        //cudaErrCheck(cudaFuncSetAttribute(
        //    opt4wl79_32x32x32<mode, block_y>, cudaFuncAttributePreferredSharedMemoryCarveout,
        //    carveout));
 
        //cudaErrCheck(cudaDeviceSetCacheConfig (cudaFuncCachePreferL1));

        //cudaErrCheck(cudaFuncSetAttribute(
        //            opt4wl79_32x32x32<mode, block_y>, 
        //            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

        opt5wl79_32x32x32<mode, block_y><<<blocks, threads>>>(in);
        //opt4wl79_32x32x32<mode, block_y><<<blocks, threads>>>(in);
        cudaErrCheck(cudaPeekAtLastError());
}

