#pragma once
#ifdef NVCC
#define __dev __device__
#define __host __host__
#else 
#define __dev 
#define __host 
#endif

#include <assert.h>

__dev __host void print_array(float *a, int nx, int ny, int nz, int ix0=0, int iy0=0, int iz0=0, int ixn=0, int iyn=0, int izn=0) {

        ixn = ixn == 0 ? nx : ixn;
        iyn = iyn == 0 ? ny : iyn;
        izn = izn == 0 ? nz : izn;

        assert(nx <= ixn);
        assert(ny <= iyn);
        assert(nz <= izn);

        for (int iz = iz0; iz < izn; ++iz) {
                printf("iz = %d \n", iz);
        for (int iy = iy0; iy < iyn; ++iy) {
                for (int ix = ix0; ix < ixn; ++ix) {
                        size_t pos = ix + nx * iy + nx * ny *  iz;
                        printf("%.2f ", a[pos]);
                }
                printf("\n");
        }
        }

}
