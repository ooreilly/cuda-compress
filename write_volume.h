void write_volume(const char *filename, const float *x, int nx, int ny, int nz, int bx, int by, int
bz) {
        FILE *fh = fopen(filename, "wb");
        fwrite(&nx, sizeof(int), 1, fh);
        fwrite(&ny, sizeof(int), 1, fh);
        fwrite(&nz, sizeof(int), 1, fh);
        fwrite(&bx, sizeof(int), 1, fh);
        fwrite(&by, sizeof(int), 1, fh);
        fwrite(&bz, sizeof(int), 1, fh);
        size_t n = nx * ny * nz * bx * by * bz;
        fwrite(x, sizeof(float), n, fh);
        fclose(fh);
}

