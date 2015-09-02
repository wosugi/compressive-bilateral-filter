# compressive-bilateral-filter



## Overview

This repository provides sample codes of the compressive bilateral filter, which is an efficient constant-time bilateral filter published on *IEEE Transactions on Image Processing* in 2015.

All the codes in this repository is open to public under the MIT license and copyrighted by Kenjiro Sugimoto. Note that the algorithm might be patented in Japan.

C++ version and Matlab version are currently offered.



## Requests

If you use some of the provided codes for research purpose, you are requested to basically cite the following two papers:

1. K. Sugimoto and S. Kamata: **"Compressive bilateral filtering"**, *IEEE Transactions on Image Processing*, vol. 24, no. 11, pp. 3357-3369 (Nov. 2015).
2. K. Sugimoto and S. Kamata: **"Efficient constant-time Gaussian filtering with sliding DCT/DST-5 and dual-domain error minimization"**, *ITE Transactions on Media Technology and Applications*, vol. 3, no. 1, pp. 12-21 (Jan. 2015).

This is because [1] employs [2] as an O(1) Gaussian spatial filter. If you do NOT use the O(1) Gaussian spatial filter, you do NOT have to cite [2]. The above two papers correspond to the following C++/Matlab files in this repository:

* [1] corresponds to **compressive_bilateral_filter.hpp** and **cbf.m**.
* [2] corresponds to **o1_spatial_gaussian_filter.hpp** and **o1filter.m/cpp**.

For COMMERCIAL purpose, it would be better to contact the authors before using these codes because some idea in the paper will be patented in Japan.



## C++ Version

Even though this version has been originally made on Microsoft Visual Studio 2010, there is no environment-dependent routine, which will be compilable on GCC etc.

### Installation

The C++ version requires the following libraries. Note that Boost is unnecessary if you use only several fixed tolerance values.

* OpenCV 2.x
* *Boost C++ Libraries (if you would like to flexibly adjust tolerance)*

The use of each library can be switched by defining `USE_OPENCV2` or `USE_BOOST` (see the head of main.cpp).

### Usage

```
cbf [InputImagePath]
```

Please change the filter parameter in the code directly if you try other parameter values.



## Matlab Version

<u>Please note that the Matlab version might contain some bugs because its approximate accuracy is slightly lower than that of the C++ version.</u>

### Installation

The Matlab version essentially requires no library. First, you have to generate an MEX file for the O(1) spatial filter by command `build_mex`.

### Usage

The program can run by command `main`. Please change the filter parameter in the code directly if you try other parameter values.


