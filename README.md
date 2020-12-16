# cuda-compress 

A CUDA version of [CvxCompress](https://github.com/ChevronETC/CvxCompress). 
I have written this code to practice my CUDA skills. The code itself is pretty much useless at the
moment and is highly under construction.

## Supported features
* 8 x 8 x 8 Forward and inverse transform.
* 32 x 32 x 32 Forward and inverse transform.


## Test with pre-generated input
Test all forward and inverse kernels with random input data
```
 ./test_all_blocks.x 
dim = [8 8 8] blocks = [11 9 8] OK
dim = [32 32 32] blocks = [11 9 8] OK
```

## Test with user-generated input
This test lets you load in a binary data array from disk. 


All tests check that the forward followed by the inverse transform recovers the identity transform. The CPU test uses a single block only. 
I mainly use this test during development. This test itself is not strong enough to guarantee the
correctness of the code. For example, if do nothing in the forward or inverse kernels, the test will
pass! I plan to add more tests.

Generate example data
```
$ ./write_volume.x x32.bin 32 32 32 4 4 4
Writing 4x4x4 blocks of dimension 32x32x32 to x32.bin
```
Run test
```
$ ./test_wavelet_transform_slow.x x32.bin 
reading: x32.bin 
block dimension: 32 32 32 
number of blocks: 4 4 4 
Computing CPU forward transform (single block) ... 
Computing CPU inverse transform (single block) ... 
abs. l2 error = 0.00130518 l1 error = 0.183796 linf error = 2.67029e-05 
rel. l2 error = 4.9955e-08 l1 error = 5.65425e-09 linf error = 8.61383e-07 
[32, 32, 32] Computing GPU forward transform... 
Throughput: 1564.81 Mcells/s 
[32, 32, 32] Computing GPU inverse transform... 
Throughput: 1672.43 Mcells/s 
Running error checking... 
abs. l2 error = 0.00981206 l1 error = 10.6313 linf error = 2.67029e-05 
rel. l2 error = 3.75552e-07 l1 error = 3.27058e-07 linf error = 8.61383e-07 
[8, 8, 8] Computing GPU forward transform... 
Throughput: 44887.7 Mcells/s 
[8, 8, 8] Computing GPU inverse transform... 
Throughput: 360088 Mcells/s 
Running error checking... 
abs. l2 error = 0.000740677 l1 error = 0.090505 linf error = 2.28882e-05 
rel. l2 error = 2.83491e-08 l1 error = 2.78427e-09 linf error = 7.38329e-07 
Test(s) passed!
```

## Throughput
Test the throughput performance of each kernel. I obtained the output below using a RTX 2080 Ti
card. 
```
./test_wavelet_transform_with_input.x 
Kernel name       	 Wavelet transform 	 Block dimension 	 Grid dimension 	 Throughput
wl79_8x8x8           	 Forward 	         [8, 8, 8] 	         [352, 416, 320] 	 66678.2 Mcells/s
wl79_8x8x8           	 Forward 	         [8, 8, 8] 	         [704, 832, 640] 	 67242 Mcells/s
wl79_8x8x8           	 Forward 	         [8, 8, 8] 	         [1056, 1248, 960] 	 67741.1 Mcells/s
opt4wl79_32x32x32    	 Forward 	         [32, 32, 32] 	         [320, 384, 416] 	 29333.4 Mcells/s
opt4wl79_32x32x32    	 Forward 	         [32, 32, 32] 	         [640, 800, 640] 	 30774.7 Mcells/s
opt4wl79_32x32x32    	 Forward 	         [32, 32, 32] 	         [1280, 1024, 576] 	 30883.3 Mcells/s
wl79_8x8x8           	 Inverse 	         [8, 8, 8] 	         [352, 416, 320] 	 52022.2 Mcells/s
wl79_8x8x8           	 Inverse 	         [8, 8, 8] 	         [704, 832, 640] 	 54296.4 Mcells/s
wl79_8x8x8           	 Inverse 	         [8, 8, 8] 	         [1056, 1248, 960] 	 54558.9 Mcells/s
opt4wl79_32x32x32    	 Inverse 	         [32, 32, 32] 	         [320, 384, 416] 	 29605.2 Mcells/s
opt4wl79_32x32x32    	 Inverse 	         [32, 32, 32] 	         [640, 800, 640] 	 30999.4 Mcells/s
opt4wl79_32x32x32    	 Inverse 	         [32, 32, 32] 	         [1280, 1024, 576] 	 31073.1 Mcells/s
```
Only the 32x32x32 kernel has been optimized at the moment. More optimizations to come...

