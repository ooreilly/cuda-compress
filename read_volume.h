#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

void read_volume(const char *filename, float*& x, int& nx, int& ny, int& nz, int& bx, int& by, int&
bz) {
        FILE *fh = fopen(filename, "rb");
        if (!fh) {
                fprintf(stderr, "Unable to open: %s \n", filename);
                return;
        }
        int count;
        count = fread(&nx, sizeof(int), 1, fh);
        assert(count == 1);
        count = fread(&ny, sizeof(int), 1, fh);
        assert(count == 1);
        count = fread(&nz, sizeof(int), 1, fh);
        assert(count == 1);
        count = fread(&bx, sizeof(int), 1, fh);
        assert(count == 1);
        count = fread(&by, sizeof(int), 1, fh);
        assert(count == 1);
        count = fread(&bz, sizeof(int), 1, fh);
        assert(count == 1);
        size_t b = bx * by * bz;
        size_t n = nx * ny * nz;
        x = (float*)malloc(sizeof(float) * n * b);
        count = fread(x, sizeof(float), n * b, fh);
        assert(count == n * b);
        fclose(fh);
}

