// Write bx * by * bz blocks each of dimension nx * ny * nz
void init_z(float*& out, int nx, int ny, int nz, int bx, int by, int bz) {

        size_t block = nx * ny * nz;
        out = (float*)malloc(sizeof(float) * block * bx * by * bz);

        // Number of blocks
        for (int jz = 0; jz < bz; ++jz)
        for (int jy = 0; jy < by; ++jy)
        for (int jx = 0; jx < bx; ++jx) {
                // Block dimension
                for (int iz = 0; iz < nz; ++iz)
                for (int iy = 0; iy < ny; ++iy)
                for (int ix = 0; ix < nx; ++ix) {
                        size_t pos_block =  ix + nx * (iy + ny * iz);
                        size_t pos = pos_block + block * (jx + bx * (jy + by * jz));
                        out[pos] = iz;
                }
        }
}

