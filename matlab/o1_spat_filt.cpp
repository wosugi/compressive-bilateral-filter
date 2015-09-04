////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015 Kenjiro Sugimoto
// Released under the MIT license
// http://opensource.org/licenses/mit-license.php
////////////////////////////////////////////////////////////////////////////////

// This code implements the algorithm of the following paper. Please cite it in 
// your paper if your research uses this code.
//   + K. Sugimoto and S. Kamata: "Efficient constant-time Gaussian filtering 
//     with sliding DCT/DST-5 and dual-domain error minimization", ITE Trans. 
//     Media Technol. Appl., vol. 3, no. 1, pp. 12-21 (Jan. 2015).

#define _USE_MATH_DEFINES
#include <mex.h>
#include <cstdio>
#include <stdexcept>
#include "../CompressiveBilateralFilter/o1_spatial_gaussian_filter.hpp"

const int K=2; // number of basis functions of spatial Gaussian kernel

// note that matlab images have been transposed in mex.
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
//	if(nrhs!=2)
//	{
//		mexErrMsgIdAndTxt("ctfilter():nrhs","2 inputs required!");
//	}
	
	// args[0] : input image
	const int dims_sz=mxGetNumberOfDimensions(prhs[0]);
	const mwSize* dims=mxGetDimensions(prhs[0]);
	double* p=reinterpret_cast<double*>(mxGetData(prhs[0]));
	
	const int h=dims[0];
	const int w=dims[1];
	const int ch=(2<dims_sz)?dims[2]:1;
	//printf("dims_sz = %d\n",dims_sz);
	//printf("Image format:  Size=(%d,%d) Ch=%d\n",w,h,ch);
	
	// args[1] : sigma
	double sigma=mxGetScalar(prhs[1]);
	//printf("Filter parameters:  sigma=%f\n",sigma);
	
	o1_spatial_gaussian_filter<K> gaussian(sigma);
	
//	if(nlhs!=1 && nlhs!=2)
//	{
//		mexErrMsgIdAndTxt("ctfilter():nlhs","1 or 2 output required!");
//	}
	
	// return[0] : output image
	plhs[0]=mxCreateNumericArray(dims_sz,dims,mxDOUBLE_CLASS,mxREAL);
	double* q=mxGetPr(plhs[0]);
	
	// return[1] : scale factor
	plhs[1]=mxCreateDoubleMatrix(1,1,mxREAL);
	double ratio=gaussian.window_size()*gaussian.window_size();
	*mxGetPr(plhs[1])=ratio;
	
	// filtering process
	for(int c=0;c<ch;++c)
		gaussian.filter_xy<double,1>(h,w,&p[c*w*h],&q[c*w*h]);
}
