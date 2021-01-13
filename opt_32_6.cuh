#include "opt32.cuh"

#pragma once
template <int kernel>
inline __device__ void opt6ds79_compute(float *p_in, int stride) {
        if (kernel == 0)
                opt5ds79_compute2(p_in, stride);
        else
                us79_compute<32>(p_in, stride);
}
 
template <int kernel, int block_y>
__launch_bounds__(32 * block_y, 1)
__global__ void opt6wl79_32x32x32(float *in) {

        int idx = threadIdx.x;
        int idy = threadIdx.y;

        const int planes = 12;
        const int block_size = 32 * 32 * 32;

        const int snx = 33;
        const int sny = 32;
        const int snxy = snx * sny;

        extern __shared__ float smem[];

        size_t block_idx =
            block_size *
            (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));

        // Load one plane per warp into registers
        float values[32];
        const int ld = 32;
        const int plane_size = 32 * 32;


        #pragma unroll
        for (int i = 0; i < ld; ++i) {
                values[i] = in[idx + i * ld + idy * plane_size + block_idx];
        }

        // Store data in shared memory in groups of 12, 12, 8, warps
        const int num_groups = 3;
        int group_size = planes;
        int group = idy / group_size;
        int group_idx = idy % group_size;

        // Transform in y
        opt6ds79_compute<kernel>(values, 1);

        for (int g = 0; g < num_groups; ++g) {
                // store in shared memory
                if (g == group) {

                        #pragma unroll
                        for (int i = 0; i < ld; ++i) {
                                int smem_idx = idx + snx * i + group_idx * snxy;
                                smem[smem_idx] = values[i];
                        }


                        __syncwarp();

                        // Transform in x
                        opt6ds79_compute<kernel>(&smem[snx * idx + group_idx * snxy], 1);

                        __syncwarp();

                        // Write result to registers
                        #pragma unroll
                        for (int i = 0; i < ld; ++i) {
                                int smem_idx = idx + snx * i + group_idx * snxy;
                                values[i] = smem[smem_idx];
                        }
                }

                __syncthreads();
        }

        // Processes along the y-direction with all warps at once, but in three batches due to
        // shared memory limitations
        int offset = 0;
        #pragma unroll
        for (int g = 0; g < num_groups; ++g) {

                #pragma unroll
                for (int i = 0; i < group_size; ++i) {
                        int smem_idx = idy + snx * idx + snxy * i;
                        if (offset + i > 31) break;
                        smem[smem_idx] = values[i + offset];
                        offset += 1;
                }


                __syncthreads();

                offset -= group_size;

                // Transform in z
                if (g == group) {
                opt6ds79_compute<kernel>(&smem[snx * idx + snxy * group_idx], 1);
                }
                
                __syncthreads();

                // Write result to registers
                #pragma unroll
                for (int i = 0; i < group_size; ++i) {
                        int smem_idx = idy + snx * idx + snxy * i;
                        values[i + offset] = smem[smem_idx];
                        offset += 1;
                }
                
                __syncthreads();
        }

        // Write final result to global memory
        #pragma unroll
        for (int i = 0; i < ld; ++i) {
                in[idx + i * ld + idy * plane_size + block_idx] = values[i];
        }

}

template <int mode>
void opt6wl79_32x32x32_h(float *in, const int bx, const int by, const int bz) {
        const int block_y = 32;
        dim3 threads(32, block_y, 1);
        dim3 blocks(bx, by, bz);
        const size_t smem = 50688;


        cudaErrCheck(cudaFuncSetAttribute(
                    opt6wl79_32x32x32<mode, block_y>, 
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

        opt6wl79_32x32x32<mode, block_y><<<blocks, threads, smem>>>(in);
        cudaErrCheck(cudaPeekAtLastError());
}

