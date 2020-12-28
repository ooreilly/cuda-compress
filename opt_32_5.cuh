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


        //ds79_compute2<size>(p_in, stride);
        opt5ds79_compute2(p_in, stride);
        //print_array(p_in, 32, 1, 1);

        //float4 p0, p1, p2, p3, p4, p5, p6, p7, p8;
        int n = 32;

       // int i0 = 0;
       //int pm1 = p_in[im1];
       //int pm2 = p_in[im2];
       //int pm3 = p_in[im3];
       //int pm4 = p_in[im4];

       //int p0 = p_in[i0];
       //int pp1 = p_in[im1];
       //int pp2 = p_in[im2];
       //int pp3 = p_in[im3];
       //int pp4 = p_in[im4];
       //
       //
       // Indices for the Left boundary
      //  4 3 2 1 0 1 2 3 4 
      //  2 1 0 1 2 3 4 5 6 

      //  0 1 2 3 4 5 6 7 8 
      //  2 3 4 5 6 7 8 9 10 
      //  4 5 6 7 8 9 10 11 12 
      //  6 7 8 9 10 11 12 13 14 
      //  8 9 10 11 12 13 14 15 16 
      //  10 11 12 13 14 15 16 17 18 
      //  12 13 14 15 16 17 18 19 20 
      //  14 15 16 17 18 19 20 21 22 
      //  16 17 18 19 20 21 22 23 24 
      //  18 19 20 21 22 23 24 25 26 
      //  20 21 22 23 24 25 26 27 28 
      //  22 23 24 25 26 27 28 29 30 

      // // Indices for the right boundary
      //  24 25 26 27 28 29 30 31 30 
      //  26 27 28 29 30 31 30 29 28

       //   int nl = 16;
       // float p_m4, p_m3, p_m2, p_m1, p_00, p_p1, p_p2, p_p3, p_p4;
       // p_m4 = p_in[4];
       // p_m3 = p_in[3];
       // p_m2 = p_in[2];
       // p_m1 = p_in[1];
       // p_00 = p_in[0];
       // p_p1 = p_in[1];
       // p_p2 = p_in[2];
       // p_p3 = p_in[3];
       // p_p4 = p_in[4];

       //for (int ix = 1;  ix < 14;  ++ix)
       //{

       //        // Low
       //        {
       //         float acc1 = al4 * (p_m4 + p_p4);
       //         acc1 += al1 * (p_m1 + p_p1);
       //         acc1 += al0 * p_00;
       //         float acc2 = al3 * (p_m3 + p_p3);
       //         acc2 += al2 * (p_m2 + p_p2);
       //         p_in[ix] = acc1 + acc2;
       //        }

       //        // High
       //        {
       // 	float acc1 = ah3 * (p_m3 + p_p3);
       // 	acc1 += ah0 * p_00;
       // 	float acc2 = ah2 * (p_m2 + p_p2);
       // 	acc2 += ah1 * (p_m1 + p_p1);
       // 	p_in[(nl+ix)*stride] = acc1 + acc2;
       //        }

       //         // Cycle registers
       //         p_m4 = p_m2;
       //         p_m3 = p_m1;
       //         p_m2 = p_00;
       //         p_m1 = p_p1;
       //         p_00 = p_p2;
       //         p_p1 = p_p3;
       //         p_p2 = p_p4;
       //         p_p3 = p_in[5 + 2 * ix];
       //         p_p4 = p_in[6 + 2 * ix + 1];
       //}

	//for (int n = size / 2;  n >= 2;  n = n-n/2)
	//{

	//	// copy inputs to tmp buffer, p_in will be overwritten
	//	for (int i = 0;  i < n;  ++i) p_tmp[i] = p_in[i*stride];
        //        //printf("n = %d, %f %f %f %f \n", n, p_tmp[0], p_tmp[1], p_tmp[2], p_tmp[3]);
        //        print_array(p_in, 32, 1, 1);


        //        printf("n = %d \n", n);

	//	int nh = n / 2;
	//	int nl = n - nh;
	//	for (int ix = 0;  ix < nl;  ++ix)
	//	{


	//		int i0 = 2 * ix;
	//		int im1 = dMIRR(i0-1,n);  int ip1 = dMIRR(i0+1,n);
	//		int im2 = dMIRR(i0-2,n);  int ip2 = dMIRR(i0+2,n);
	//		int im3 = dMIRR(i0-3,n);  int ip3 = dMIRR(i0+3,n);
	//		int im4 = dMIRR(i0-4,n);  int ip4 = dMIRR(i0+4,n);

	//		// sum smallest to largest (most accurate way of summing floats)
	//		float acc1 = al4 * (p_tmp[im4] + p_tmp[ip4]);
	//		acc1 += al1 * (p_tmp[im1] + p_tmp[ip1]);
	//		acc1 += al0 * p_tmp[i0];
	//		float acc2 = al3 * (p_tmp[im3] + p_tmp[ip3]);
	//		acc2 += al2 * (p_tmp[im2] + p_tmp[ip2]);
	//		p_in[ix*stride] = acc1 + acc2;
	//	}
	//	for (int ix = 0;  ix < nh;  ++ix)
	//	{
	//		int i0 = 2 * ix + 1;
	//		int im1 = dMIRR(i0-1,n);  int ip1 = dMIRR(i0+1,n);
	//		int im2 = dMIRR(i0-2,n);  int ip2 = dMIRR(i0+2,n);
	//		int im3 = dMIRR(i0-3,n);  int ip3 = dMIRR(i0+3,n);

	//		float acc1 = ah3 * (p_tmp[im3] + p_tmp[ip3]);
	//		acc1 += ah0 * p_tmp[i0];
	//		float acc2 = ah2 * (p_tmp[im2] + p_tmp[ip2]);
	//		acc2 += ah1 * (p_tmp[im1] + p_tmp[ip1]);
	//		p_in[(nl+ix)*stride] = acc1 + acc2;
	//	}
	//}

}

// Reorganize loads
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
        const int block_y = 8;
        dim3 threads(32, block_y, 1);
        dim3 blocks(bx, by, bz);
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

