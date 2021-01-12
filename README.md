# cuda-compress 

A CUDA version of [CvxCompress](https://github.com/ChevronETC/CvxCompress). 
I have written this code to practice my CUDA skills. The code itself is pretty much useless at the
moment and is highly under construction.

## Supported features
* 8 x 8 x 8 Forward and inverse transform.
* 32 x 32 x 32 Forward and inverse transform. 

**NOTE**: Only the forward transform has been optimized.


## Test with pre-generated input
Test all forward and inverse kernels with random input data
```
 ./test_all_blocks.x 
wl79_8x8x8 	 [8, 8, 8] [11, 9, 8] 
OK
wl79_32x32x32 	 [32, 32, 32] [11, 9, 8] 
OK
opt1wl79_32x32x32 	 [32, 32, 32] [11, 9, 8] 
OK
opt2wl79_32x32x32 	 [32, 32, 32] [11, 9, 8] 
OK
opt3wl79_32x32x32 	 [32, 32, 32] [11, 9, 8] 
OK
opt4wl79_32x32x32 	 [32, 32, 32] [11, 9, 8] 
OK
opt5wl79_32x32x32 	 [32, 32, 32] [11, 9, 8] 
OK
opt6wl79_32x32x32 	 [32, 32, 32] [11, 9, 8] 
FAILED
opt7wl79_32x32x32 	 [32, 32, 32] [11, 9, 8] 
OK
```
Optimization version 6 does not work and it is only not very fast :). 

## Test with user-generated input
This test lets you load in a binary data array from disk. 


All tests check that the forward followed by the inverse transform recovers the identity transform. The CPU test uses a single block only. 
I mainly use this test during development. This test itself is not strong enough to guarantee the
correctness of the code. For example, if the computation does nothing in both the forward or inverse kernels, the test will
pass! I plan to add more tests.

Generate example data
```
$ ./write_volume.x rand32.bin 32 32 32 1 2 3 3
Generating random grid 
Writing 1x2x3 blocks of dimension 32x32x32 to rand32.bin 
```
Run test
```
$ ./test_wavelet_transform_slow.x x32.bin 
reading: rand32.bin 
block dimension: 32 32 32 
number of blocks: 1 2 3 
[32, 32, 32] Computing GPU forward transform... 
Throughput: 4788.78 Mcells/s 
[32, 32, 32] Computing GPU inverse transform... 
Throughput: 263.827 Mcells/s 
Running error checking... 
abs. l2 error = 0.000101344 l1 error = 0.0368315 linf error = 9.53674e-07 
rel. l2 error = 3.96186e-07 l1 error = 3.74868e-07 linf error = 9.5368e-07 
```

## Throughput
Test the throughput performance of each kernel. I obtained the output below using a RTX 2080 Ti
card. 
```
./test_wavelet_transform_with_input.x 
Kernel name       	 Wavelet transform 	 Block dimension 	 Grid dimension 	 Throughput
wl79_32x32x32        	 Forward 	         [32, 32, 32] 	         [320, 384, 416] 	 1968.43 Mcells/s
opt1wl79_32x32x32    	 Forward 	         [32, 32, 32] 	         [320, 384, 416] 	 8027.09 Mcells/s
opt2wl79_32x32x32    	 Forward 	         [32, 32, 32] 	         [320, 384, 416] 	 15068.8 Mcells/s
opt3wl79_32x32x32    	 Forward 	         [32, 32, 32] 	         [320, 384, 416] 	 19548.7 Mcells/s
opt4wl79_32x32x32    	 Forward 	         [32, 32, 32] 	         [320, 384, 416] 	 29149.3 Mcells/s
opt5wl79_32x32x32    	 Forward 	         [32, 32, 32] 	         [320, 384, 416] 	 35942 Mcells/s
opt6wl79_32x32x32    	 Forward 	         [32, 32, 32] 	         [320, 384, 416] 	 33031.6 Mcells/s
opt7wl79_32x32x32    	 Forward 	         [32, 32, 32] 	         [320, 384, 416] 	 54877.9 Mcells/s
```
Only the 32x32x32 kernel has been optimized at the moment. Only the forward transform has been
optimized for versions 5 and up. Version 6 doesn't work yet. More optimizations to come...

