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
This test lets you load in a binary data array from disk. The test runs a forward and inverse pass.
I mainly use this test during development.

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
Wavelet transform 	 Block dimension 	 Grid dimension 	 Throughput
Forward 	         [8, 8, 8] 	         [352, 416, 320] 	 67183 Mcells/s
Inverse 	         [8, 8, 8] 	         [352, 416, 320] 	 53651.9 Mcells/s
Forward 	         [8, 8, 8] 	         [704, 832, 640] 	 67408 Mcells/s
Inverse 	         [8, 8, 8] 	         [704, 832, 640] 	 54031.7 Mcells/s
Forward 	         [8, 8, 8] 	         [1056, 1248, 960] 	 67699.4 Mcells/s
Inverse 	         [8, 8, 8] 	         [1056, 1248, 960] 	 50154.9 Mcells/s
Forward 	         [32, 32, 32] 	         [320, 384, 416] 	 1974.3 Mcells/s
Inverse 	         [32, 32, 32] 	         [320, 384, 416] 	 1740.14 Mcells/s
Forward 	         [32, 32, 32] 	         [640, 800, 640] 	 1989.96 Mcells/s
Inverse 	         [32, 32, 32] 	         [640, 800, 640] 	 1926.86 Mcells/s
Forward 	         [32, 32, 32] 	         [1280, 1024, 576] 	 1996.37 Mcells/s
Inverse 	         [32, 32, 32] 	         [1280, 1024, 576] 	 1967.1 Mcells/s
```
As you can see, I got some more work to do for the bigger block sizes :).

