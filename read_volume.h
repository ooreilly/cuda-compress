#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

void read_volume(const char *filename, float*& x, int& nx, int& ny, int& nz) {
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
        size_t n = nx * ny * nz;
        x = (float*)malloc(sizeof(float) * n);
        count = fread(x, sizeof(float), n, fh);
        assert(count == n);
        fclose(fh);
}

