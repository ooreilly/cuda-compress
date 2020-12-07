#pragma once
#define al0  8.526986790094000e-001f
#define al1  3.774028556126500e-001f
#define al2 -1.106244044184200e-001f
#define al3 -2.384946501938001e-002f
#define al4  3.782845550699501e-002f

#define ah0  7.884856164056601e-001f
#define ah1 -4.180922732222101e-001f
#define ah2 -4.068941760955800e-002f
#define ah3  6.453888262893799e-002f

inline int MIRR(int inp_val, int dim)
{
	int val = inp_val < 0 ? -inp_val : inp_val;
	val = (val >= dim) ? (2*dim-2-val) : val;
	val = val < 0 ? -val : val;
	val = (val >= dim) ? (2*dim-2-val) : val;
	//printf("  -> -> MIRR(%d,%d) = %d\n",inp_val,dim,val);
	return val;
}

inline void
Ds79(
	float* p_in,
	float* p_tmp,
	int stride,
	int dim
	)
{
		//printf("  -> n=%d, nh=%d, nl=%d\n",n,nh,nl);
                //
	for (int n = dim;  n >= 2;  n = n-n/2)
	{

		// copy inputs to tmp buffer, p_in will be overwritten
		for (int i = 0;  i < n;  ++i) p_tmp[i] = p_in[i*stride];

		int nh = n / 2;
		int nl = n - nh;
		for (int ix = 0;  ix < nl;  ++ix)
		{

			int i0 = 2 * ix;
			int im1 = MIRR(i0-1,n);  int ip1 = MIRR(i0+1,n);
			int im2 = MIRR(i0-2,n);  int ip2 = MIRR(i0+2,n);
			int im3 = MIRR(i0-3,n);  int ip3 = MIRR(i0+3,n);
			int im4 = MIRR(i0-4,n);  int ip4 = MIRR(i0+4,n);

			// sum smallest to largest (most accurate way of summing floats)
			float acc1 = al4 * (p_tmp[im4] + p_tmp[ip4]);
			acc1 += al1 * (p_tmp[im1] + p_tmp[ip1]);
			acc1 += al0 * p_tmp[i0];
			float acc2 = al3 * (p_tmp[im3] + p_tmp[ip3]);
			acc2 += al2 * (p_tmp[im2] + p_tmp[ip2]);
			p_in[ix*stride] = acc1 + acc2;
		}
		//for (int ix = 0;  ix < nh;  ++ix)
		//{
		//	int i0 = 2 * ix + 1;
		//	int im1 = MIRR(i0-1,n);  int ip1 = MIRR(i0+1,n);
		//	int im2 = MIRR(i0-2,n);  int ip2 = MIRR(i0+2,n);
		//	int im3 = MIRR(i0-3,n);  int ip3 = MIRR(i0+3,n);

		//	float acc1 = ah3 * (p_tmp[im3] + p_tmp[ip3]);
		//	acc1 += ah0 * p_tmp[i0];
		//	float acc2 = ah2 * (p_tmp[im2] + p_tmp[ip2]);
		//	acc2 += ah1 * (p_tmp[im1] + p_tmp[ip1]);
		//	p_in[(nl+ix)*stride] = acc1 + acc2;
		//}
	}
}

template <int Y, int Z>
__global__ void ds79_32xYxZ(float *in, float *tmp, int stride, int nx, int ny, int nz) {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int idy = threadIdx.y + blockDim.y * blockIdx.y;

        __shared__ float *smem;
        const int warp_size = 32;

        size_t line = nx;
        size_t slice = line * ny;

        // Load 2d plane into shared memory
        size_t sptr = idx + warp_size * idy;
        smem[sptr] = in[x + y * line + z * slice]; 


}

