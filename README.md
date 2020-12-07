# cuda-compress 

A CUDA version of [CvxCompress](https://github.com/ChevronETC/CvxCompress). 
I have written this code to practice my CUDA skills. The code itself is pretty much useless at the
moment and is highly under construction.

## Supported features:
* 8 x 8 x 8 Forward and inverse transform of a single block.

## Expected output
Generate example data
```
$ ./write_volume.x x888.bin 8 8 8 10 10 10
```
Run test
```
$ ./test_wavelet_transform_slow.x x888.bin
 reading: x888.bin 
block dimension: 8 8 8 
number of blocks: 10 10 10 
Computing CPU forward transform... 
Computing CPU inverse transform... 
l2 error = 3.76287e-05 l1 error = 0 linf error = 3.8147e-06 
Computing GPU forward transform... 
Computing GPU inverse transform... 
abs. l2 error = 0.00114194 l1 error = 2.86102e-06 linf error = 4.29153e-06 
rel. l2 error = 3.81497e-07 l1 error = 1.59655e-12 linf error = 6.13076e-07
```

