////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015 Kenjiro Sugimoto
// Released under the MIT license
// http://opensource.org/licenses/mit-license.php
////////////////////////////////////////////////////////////////////////////////
#define _USE_MATH_DEFINES
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "original_bilateral_filter.hpp"
#include "compressive_bilateral_filter.hpp"

#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_contrib"CV_VERSION_NUMBER"d.lib")
#else
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_contrib"CV_VERSION_NUMBER".lib")
#endif

// parameters of BF algorithms
const double sigmaS=2.0;
const double sigmaR=0.1;
const double tau=0.1; // for compressive BF
const int tone=(1<<8); // assuming 8-bits dynamic range, ie, [0,256).

const bool sw_display_results=false;

template<typename T>
double calc_snr(const cv::Mat_<T>& image1,const cv::Mat_<T>& image2,T minval,T maxval)
{
	assert(image1.size()==image2.size());
	
	double sse=0.0;
	for(int y=0;y<image1.rows;++y)
	for(int x=0;x<image1.cols;++x)
	{
		T p=image1(y,x);
		T q=image2(y,x);
		p=(p<minval)?minval:(maxval<p)?maxval:p;
		q=(q<minval)?minval:(maxval<q)?maxval:q;
		sse+=(q-p)*(q-p);
	}
	const double EPS=0.000001;
	if(sse<=EPS)
		return 0.0; //means the infinity

	double mse=sse/(image1.cols*image1.rows);
	double snr=-10.0*log10(mse/((maxval-minval)*(maxval-minval)));
	return snr;
}

int main(int argc,char** argv)
{
	if(argc!=2)
	{
		std::cerr<<"Usage: cbf [ImageInputPath]"<<std::endl;
		return 1;
	}

	const std::string& pathI(argv[1]);
	cv::Mat image0=cv::imread(pathI,-1); // grayscale only
	if(image0.empty())
	{
		std::cerr<<"Input image loading failed!"<<std::endl;
		return 1;
	}
	if(image0.channels()!=1)
	{
		std::cerr<<"Input image should be with 1 channel!"<<std::endl;
		return 1;
	}
	
	cv::Mat_<double> image=image0/double(tone-1); // dynamic range is transformed to [0,1]
	cv::Mat_<double> dst0(image.size());
	cv::Mat_<double> dst1(image.size());

	std::cerr<<cv::format("[ss=%f sr=%f]",sigmaS,sigmaR)<<std::endl;

	cv::TickMeter tm;
	
	tm.start();
	apply_bilateral_filter_original(image,dst0,sigmaS,sigmaR,tone);
	tm.stop();
	std::cerr<<cv::format("Original BF:     %7.1f [ms]",tm.getTimeMilli())<<std::endl;
	tm.reset();

	tm.start();
	compressive_bilateral_filter cbf(sigmaS,sigmaR,tau);
	cbf(image,dst1,tone);
	tm.stop();
	std::cerr<<cv::format("Compressive BF:  %7.1f [ms]",tm.getTimeMilli())<<std::endl;
	tm.reset();

	double snr=calc_snr<double>(dst0,dst1,0.0,1.0);
	std::cerr<<cv::format("SNR:  %f",snr)<<std::endl;

	if(sw_display_results)
	{
		//cv::imwrite(pathO,dst1*(tone-1)); // dynamic range is transformed back to [0,tone)
		//cv::imshow("src",image);
		cv::imshow("dst0",dst0);
		cv::imshow("dst1",dst1);
		cv::waitKey();
	}
	return 0;
}
