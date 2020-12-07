#pragma once

template <typename T>
int compare(T *a, T *b, int nx, int ny, int nz, int verbose=0, T tol=1e-5f) {
        int pass = 1;
        for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) {
                size_t pos = x + nx * y + nx * ny * z;
                T ax = a[pos];
                T bx = b[pos];
                T diff = ax > bx ? ax - bx : bx - ax;
                if (diff > tol) {
                        pass = 0;
                        if (verbose) printf("[%d %d %d] a = %f b = %f \n", x, y, z, ax, bx);
                }
        }

        return pass;
}
