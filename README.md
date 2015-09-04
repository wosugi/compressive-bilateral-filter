# compressive-bilateral-filter


## Overview

This repository provides sample codes of the compressive bilateral filter, which is an efficient constant-time bilateral filter published on *IEEE Transactions on Image Processing* in 2015.

All the codes in this repository is open to public under [the MIT license and copyrighted by Kenjiro Sugimoto](./LICENSE). Note that the algorithm might be patented in Japan.

C++ version and Matlab version are currently offered. You can test bilateral filtering and its cross/joint extension.



## Requests

If you use some of the provided codes for research purpose, you are requested to basically cite the following two papers:

1. K. Sugimoto and S. Kamata: **"[Compressive bilateral filtering](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7120121)"**, *IEEE Trans. Image Process.*, vol. 24, no. 11, pp. 3357--3369, (Nov. 2015).
2. K. Sugimoto and S. Kamata: **"[Efficient constant-time Gaussian filtering with sliding DCT/DST-5 and dual-domain error minimization](https://www.jstage.jst.go.jp/article/mta/3/1/3_12/_article)"**, *ITE Trans. Media Technol. Appl.*, vol. 3, no. 1, pp. 12--21, (Jan. 2015).

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
# Bilateral filter
cbf [SourceImagePath]
# Cross/Joint bilateral filter
cbf [SourceImagePath] [GuideImagePath]
```
Please directly change the filter parameter in the code (the head of main()) if you try other parameter values.



## Matlab Version

*(Please note that the Matlab version might contain some bugs because its PSNR is about 2--4dB lower than that of the C++ version.)*

### Installation

The Matlab version currently requires "Image Processing Toolbox", "Optimization Toolbox", and an MEX building environment. The O(1) spatial filter is provided as an MEX routine, which will be built during the first execution.

### Usage

The program can run by command `main`. The command will automatically try to generate the MEX routine if it is missing. Please change the filter parameter in the code directly if you use other parameter values.



## Related Literature

#### Bilateral filter

+ C. Tomasi and R. Manduchi: **"Bilateral filtering for gray and color
images"**, in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*,
pp. 839--846, (Jan. 1998).

#### Cross/Joint bilateral filter

+ G. Petschnigg, R. Szeliski, M. Agrawala, M. Cohen, H. Hoppe,
and K. Toyama: **"Digital photography with flash and no-flash
image pairs"**, *ACM Trans. Graph.*, vol. 23, no. 3, pp. 664--672,
(Aug. 2004).
+ E. Eisemann and F. Durand: **"Flash photography enhancement via
intrinsic relighting"**, *ACM Trans. Graph.*, vol. 23, no. 3, pp. 673--678,
(Aug. 2004).
