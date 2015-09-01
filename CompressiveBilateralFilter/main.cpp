////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015 Kenjiro Sugimoto
// Released under the MIT license
// http://opensource.org/licenses/mit-license.php
////////////////////////////////////////////////////////////////////////////////
#define _USE_MATH_DEFINES
#define USE_OPENCV2
//#define USE_BOOST
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

// parameters of BF algorithms (assuming 8-bits dynamic range)
const double sigmaS=2.0;
const double sigmaR=0.1*255.0;
const double tol=0.1; // for compressive BF

const bool sw_imshow=true;
//const bool sw_imwrite=false;

double calc_snr(const cv::Mat& image1,const cv::Mat& image2,double minval,double maxval)
{
	assert(image1.size()==image2.size());
	assert(image1.type()==image2.type());
	
	cv::Mat err;
	cv::absdiff(image1,image2,err);
	err.convertTo(err,CV_64F);
	err=err.mul(err);
	
	cv::Scalar sums=cv::sum(err);
	double sse=0.0;
	for(int c=0;c<sums.channels;++c)
		sse+=sums.val[c];

	const double EPS=0.000001;
	if(sse<=EPS)
		return 0.0; // zero means the infinity

	double mse=sse/(image1.total()*image1.channels());
	double snr=10.0*log10((maxval-minval)*(maxval-minval)/mse);
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
	//const std::string& pathI("../lenna-gray.png"); // for debug
	cv::Mat image0=cv::imread(pathI,-1);
	if(image0.empty())
	{
		std::cerr<<"Input image loading failed!"<<std::endl;
		return 1;
	}
	
	std::cerr<<"[Input Image]"<<std::endl;
	std::cerr<<cv::format("\"%s\"  # (w,h,ch)=(%d,%d,%d)",pathI.c_str(),image0.cols,image0.rows,image0.channels())<<std::endl;
	std::cerr<<"[Filter Parameters]"<<std::endl;
	std::cerr<<cv::format("sigmaS=%f  sigmaR=%f  tol=%f",sigmaS,sigmaR,tol)<<std::endl;
	
	cv::Mat src;
	image0.convertTo(src,CV_64F);
	std::vector<cv::Mat_<double> > srcsp;
	cv::split(src,srcsp);

	std::vector<cv::Mat_<double> > dstsp0(src.channels());
	std::vector<cv::Mat_<double> > dstsp1(src.channels());
	for(int c=0;c<int(src.channels());++c)
	{
		dstsp0[c]=cv::Mat_<double>(src.size());
		dstsp1[c]=cv::Mat_<double>(src.size());
	}

	cv::TickMeter tm;
	// Original bilateral filtering
	tm.start();
	for(int c=0;c<int(src.channels());++c)
		apply_bilateral_filter_original(srcsp[c],dstsp0[c],sigmaS,sigmaR);
	tm.stop();
	std::cerr<<cv::format("Original BF:     %7.1f [ms]",tm.getTimeMilli())<<std::endl;
	tm.reset();
	// Compressive bilateral filtering
	tm.start();
	compressive_bilateral_filter cbf(sigmaS,sigmaR,tol);
	for(int c=0;c<int(src.channels());++c)
		cbf(srcsp[c],dstsp1[c]);
	tm.stop();
	std::cerr<<cv::format("Compressive BF:  %7.1f [ms]",tm.getTimeMilli())<<std::endl;
	tm.reset();
	
	//// clipping for debug
	//double minval=  0.0;
	//double maxval=255.0;
	//for(int c=0;c<int(src.channels());++c)
	//{
	//	for(int y=0;y<src.rows;++y)
	//	for(int x=0;x<src.cols;++x)
	//	{
	//		double p0=dstsp0[c](y,x);
	//		double p1=dstsp1[c](y,x);
	//		dstsp0[c](y,x)=(p0<minval)?minval:(maxval<p0)?maxval:p0;
	//		dstsp1[c](y,x)=(p1<minval)?minval:(maxval<p1)?maxval:p1;
	//	}
	//}

	cv::Mat dst0,dst1;	
	cv::merge(dstsp0,dst0);
	cv::merge(dstsp1,dst1);

	double snr=calc_snr(dst0,dst1,0.0,255.0);
	std::cerr<<cv::format("SNR:  %f",snr)<<std::endl;
	
	if(sw_imshow)
	{
		//cv::imshow("src",image);
		cv::imshow("dst0",dst0/255.0);
		cv::imshow("dst1",dst1/255.0);
		//cv::imshow("error",(dst1-dst0)/255.0+0.5);
		cv::waitKey();
	}
	//if(sw_imwrite)
	//{
	//	cv::imwrite("../dst0.png",dst0*255.0);
	//	cv::imwrite("../dst1.png",dst1*255.0);
	//	cv::imwrite("../error.png",(dst1-dst0)+128.0);
	//}
	return 0;
}
