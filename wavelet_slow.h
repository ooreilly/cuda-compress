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

#define sl0  7.884856164056601e-001f
#define sl1  4.180922732222101e-001f
#define sl2 -4.068941760955800e-002f
#define sl3 -6.453888262893799e-002f

#define sh0  8.526986790094000e-001f
#define sh1 -3.774028556126500e-001f
#define sh2 -1.106244044184200e-001f
#define sh3  2.384946501938001e-002f
#define sh4  3.782845550699501e-002f
#define Verbose 0
inline int MIRR(int inp_val, int dim)
{
	int val = inp_val < 0 ? -inp_val : inp_val;
	val = (val >= dim) ? (2*dim-2-val) : val;
	val = val < 0 ? -val : val;
	val = (val >= dim) ? (2*dim-2-val) : val;
	//printf("  -> -> MIRR(%d,%d) = %d\n",inp_val,dim,val);
	return val;
}


inline int MIRR_SL(int inp_val, int nl)
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


inline int MIRR_SH(int inp_val, int nl, int nh)
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
		for (int ix = 0;  ix < nh;  ++ix)
		{
			int i0 = 2 * ix + 1;
			int im1 = MIRR(i0-1,n);  int ip1 = MIRR(i0+1,n);
			int im2 = MIRR(i0-2,n);  int ip2 = MIRR(i0+2,n);
			int im3 = MIRR(i0-3,n);  int ip3 = MIRR(i0+3,n);

			float acc1 = ah3 * (p_tmp[im3] + p_tmp[ip3]);
			acc1 += ah0 * p_tmp[i0];
			float acc2 = ah2 * (p_tmp[im2] + p_tmp[ip2]);
			acc2 += ah1 * (p_tmp[im1] + p_tmp[ip1]);
			p_in[(nl+ix)*stride] = acc1 + acc2;
		}
	}
}

inline void
Us79(
	float* p_in,
	float* t,
	int stride,
	int dim
	)
{
	if (Verbose) printf("Us79(*,*,stride=%d,dim=%d)\n",stride,dim);
	int* l = new int[dim];
	int nx = 0;
	for (int n = dim;  n >= 2;  n = n-n/2) {l[nx++] = n;}
	for (int li = nx-1;  li >= 0;  --li)
	{
		int n = l[li];

		if (Verbose)
		{
			printf("Us d_inp = %e",p_in[0]);
			for (int i = 1;  i < n;  ++i) printf(", %e",p_in[i*stride]);
			printf("\n");
		}

		// copy inputs to tmp buffer, p_in will be overwritten
		for (int i = 0;  i < n;  ++i) t[i] = p_in[i*stride];

		int nh = n / 2;
		int nl = n - nh;
		//printf("  -> n=%d, nh=%d, nl=%d\n",n,nh,nl);
		for (int k = 0;  k < nl;  ++k)
		{
			if (Verbose) printf("d[%d] = sl0*t[%d] + sl2*(t[%d]+t[%d]) + sh1*(t[%d]+t[%d]) + sh3*(t[%d]+t[%d])\n",2*k,k,MIRR_SL(k-1,nl),MIRR_SL(k+1,nl),MIRR_SH(nl+k-1,nl,nh),MIRR_SH(nl+k,nl,nh),MIRR_SH(nl+k-2,nl,nh),MIRR_SH(nl+k+1,nl,nh));
			p_in[2*k*stride] = 
				sl0 * t[k] + 
				sl2 * ( t[MIRR_SL(k-1,nl)] + t[MIRR_SL(k+1,nl)] ) +
				sh1 * ( t[MIRR_SH(nl+k-1,nl,nh)] + t[MIRR_SH(nl+k,nl,nh)] ) +
				sh3 * ( t[MIRR_SH(nl+k-2,nl,nh)] + t[MIRR_SH(nl+k+1,nl,nh)] );
		}
		for (int k = 0;  k < nh;  ++k)
		{
			if (Verbose) printf("d[%d] = sl1*(t[%d]+t[%d]) + sl3*(t[%d]+t[%d]) + sh0*t[%d] + sh2*(t[%d]+t[%d]) + sh4*(t[%d]+t[%d])\n",(2*k+1),MIRR_SL(k,nl),MIRR_SL(k+1,nl),MIRR_SL(k-1,nl),MIRR_SL(k+2,nl),nl+k,MIRR_SH(nl+k-1,nl,nh),MIRR_SH(nl+k+1,nl,nh),MIRR_SH(nl+k-2,nl,nh),MIRR_SH(nl+k+2,nl,nh));
			p_in[(2*k+1)*stride] = 
				sl1 * ( t[MIRR_SL(k,nl)] + t[MIRR_SL(k+1,nl)] ) +
				sl3 * ( t[MIRR_SL(k-1,nl)] + t[MIRR_SL(k+2,nl)] ) +
				sh0 * t[nl+k] +
				sh2 * ( t[MIRR_SH(nl+k-1,nl,nh)] + t[MIRR_SH(nl+k+1,nl,nh)] ) +
				sh4 * ( t[MIRR_SH(nl+k-2,nl,nh)] + t[MIRR_SH(nl+k+2,nl,nh)] );
		}

		if (Verbose)
		{
			printf("Us d_out = %e",p_in[0]);
			for (int i = 1;  i < n;  ++i) printf(", %e",p_in[i*stride]);
			printf("\n");
		}
	}
	delete [] l;
	if (Verbose) printf("\n");
}

void Wavelet_Transform_Slow_Forward(
	float* data,
	float* work,
	int bx,
	int by,
	int bz,
	int x0,
	int y0,
	int z0,
	int nx,
	int ny,
	int nz
	)
{
	for (int iz = 0;  iz < bz;  ++iz) {
		if (bx > 1) for (int iy = 0;  iy < by;  ++iy) Ds79(data+((iz+z0)*ny+(iy+y0))*nx+(x0), work,  1, bx);
		if (by > 1) for (int ix = 0;  ix < bx;  ++ix) Ds79(data+((iz+z0)*ny+(y0))*nx+(ix+x0), work, bx, by);
	}
	if (bz > 1) for (int iy = 0;  iy < by;  ++iy) for (int ix = 0;  ix < bx;  ++ix) Ds79(data+((z0)*ny+(iy+y0))*nx+(ix+x0), work, bx*by, bz);
}

void Wavelet_Transform_Slow_Inverse(
	float* data,
	float* work,
	int bx,
	int by,
	int bz,
	int x0,
	int y0,
	int z0,
	int nx,
	int ny,
	int nz
	)
{
	for (int iz = 0;  iz < bz;  ++iz) {
		if (bx > 1) for (int iy = 0;  iy < by;  ++iy) Us79(data+((iz+z0)*ny+(iy+y0))*nx+(x0), work,  1, bx);
		if (by > 1) for (int ix = 0;  ix < bx;  ++ix) Us79(data+((iz+z0)*ny+(y0))*nx+(ix+x0), work, bx, by);
	}
	if (bz > 1) for (int iy = 0;  iy < by;  ++iy) for (int ix = 0;  ix < bx;  ++ix) Us79(data+((z0)*ny+(iy+y0))*nx+(ix+x0), work, bx*by, bz);
}
