CC = gcc
CXX = g++
NVCC = nvcc
NVFLAGS=-Xptxas -v -g -use_fast_math
CFLAGS=-fopenmp -O3 -fPIC -mavx -g
LDFLAGS=-fopenmp -lm -lrt
arch=sm_75

all: write_volume read_volume test_wavelet_transform_slow test_wavelet_transform_with_input

write_volume:
	$(CXX) $(LDFLAGS) write_volume.c -o write_volume.x

read_volume:
	$(CXX) $(LDFLAGS) read_volume.c -o read_volume.x

test_wavelet_transform_slow:
	$(NVCC) $(NVFLAGS) test_wavelet_transform_slow.cu -o test_wavelet_transform_slow.x

test_wavelet_transform_with_input:
	$(NVCC) $(NVFLAGS) test_wavelet_transform_with_input.cu -o test_wavelet_transform_with_input.x

%.o: %.c
	$(CC) -c $(CFLAGS) $*.c

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $*.cpp

clean:
	rm -f *.o
	rm -f *.x
	rm -f *.bin
