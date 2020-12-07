void write_volume(const char *filename, const float *x, int nx, int ny, int nz) {
        FILE *fh = fopen(filename, "wb");
        fwrite(&nx, sizeof(int), 1, fh);
        fwrite(&ny, sizeof(int), 1, fh);
        fwrite(&nz, sizeof(int), 1, fh);
        size_t n = nx * ny * nz;
        fwrite(x, sizeof(float), n, fh);
        fclose(fh);
}

