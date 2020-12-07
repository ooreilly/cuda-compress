#include <stdio.h>
#include <stdlib.h>

#include "printing.cuh"
#include "read_volume.h"



int main(int argc, char **argv) {

        
        const char *filename = argv[1];

        float *x;
        printf("reading: %s \n", filename);
        int nx, ny, nz;
        read_volume(filename, x, nx, ny, nz);
        print_array(x, nx, ny, nz);

        free(x);
        
}

