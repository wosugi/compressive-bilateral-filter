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

// assuming 8-bits dynamic range
const double tone=256.0;

const bool sw_imshow=true;
const bool sw_imwrite=false;

double calc_psnr(const cv::Mat& image1,const cv::Mat& image2,double maxval)
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
	double snr=10.0*log10(maxval*maxval/mse);
	return snr;
}

template<typename T>
void clip(cv::Mat& image,double minval=0.0,double maxval=255.0)
{
	for(int y=0;y<image.rows;++y)
	{
		T* p=image.ptr<T>(y);
		for(int x=0;x<image.cols;++x)
		{
			for(int c=0;c<image.channels();++c)
			{
				const T value=*p;
				*p++=(value<minval)?minval:(maxval<value)?maxval:value;
			}
		}
	}
}

int test_bilateral_filter(const std::string& pathS,double sigmaS,double sigmaR,double tol)
{
	cv::Mat imageS=cv::imread(pathS,-1);
	if(imageS.empty())
	{
		std::cerr<<"Source image loading failed!"<<std::endl;
		return 1;
	}
	
	std::cerr<<"[Loaded Image]"<<std::endl;
	std::cerr<<cv::format("Source: \"%s\"  # (w,h,ch)=(%d,%d,%d)",pathS.c_str(),imageS.cols,imageS.rows,imageS.channels())<<std::endl;
	std::cerr<<"[Filter Parameters]"<<std::endl;
	std::cerr<<cv::format("sigmaS=%f  sigmaR=%f  tolerance=%f",sigmaS,sigmaR,tol)<<std::endl;
	
	cv::Mat src;
	imageS.convertTo(src,CV_64F);

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
	
	cv::Mat dst0,dst1;
	cv::merge(dstsp0,dst0);
	cv::merge(dstsp1,dst1);
//	clip(dst0); // for debug
//	clip(dst1); // for debug

	double psnr=calc_psnr(dst0,dst1,tone-1.0);
	std::cerr<<cv::format("PSNR:  %f",psnr)<<std::endl;
	
	if(sw_imshow)
	{
		//cv::imshow("src",src);
		cv::imshow("dst0",dst0/(tone-1.0));
		cv::imshow("dst1",dst1/(tone-1.0));
		//cv::imshow("error",(dst1-dst0)/(tone-1.0)+0.5);
		cv::waitKey();
	}
	if(sw_imwrite)
	{
		//cv::imwrite("../dst0.png",dst0*(tone-1.0));
		//cv::imwrite("../dst1.png",dst1*(tone-1.0));
		//cv::imwrite("../error.png",(dst1-dst0)+tone/2.0);
	}
	return 0;
}

int test_crossjoint_bilateral_filter(const std::string& pathS,const std::string& pathG,double sigmaS,double sigmaR,double tol)
{
	cv::Mat imageS=cv::imread(pathS,-1);
	if(imageS.empty())
	{
		std::cerr<<"Source image loading failed!"<<std::endl;
		return 1;
	}
	cv::Mat imageG=cv::imread(pathG,-1);
	if(imageG.empty())
	{
		std::cerr<<"Guide image loading failed!"<<std::endl;
		return 1;
	}

	std::cerr<<"[Loaded Image]"<<std::endl;
	std::cerr<<cv::format("Source: \"%s\"  # (w,h,ch)=(%d,%d,%d)",pathS.c_str(),imageS.cols,imageS.rows,imageS.channels())<<std::endl;
	std::cerr<<cv::format("Guide:  \"%s\"  # (w,h,ch)=(%d,%d,%d)",pathG.c_str(),imageG.cols,imageG.rows,imageG.channels())<<std::endl;
	if(imageS.size()!=imageG.size() || imageS.channels()!=imageG.channels())
	{
		std::cerr<<"Source and guide images should share the same size and channels!"<<std::endl;
		return 1;
	}

	std::cerr<<"[Filter Parameters]"<<std::endl;
	std::cerr<<cv::format("sigmaS=%f  sigmaR=%f  tolerance=%f",sigmaS,sigmaR,tol)<<std::endl;
	
	cv::Mat src;
	imageS.convertTo(src,CV_64F);
	cv::Mat guide;
	imageG.convertTo(guide,CV_64F);
	
	std::vector<cv::Mat_<double> > srcsp;
	cv::split(src,srcsp);
	std::vector<cv::Mat_<double> > guidsp;
	cv::split(guide,guidsp);

	std::vector<cv::Mat_<double> > dstsp0(src.channels());
	std::vector<cv::Mat_<double> > dstsp1(src.channels());
	for(int c=0;c<int(src.channels());++c)
	{
		dstsp0[c]=cv::Mat_<double>(src.size());
		dstsp1[c]=cv::Mat_<double>(src.size());
	}

	cv::TickMeter tm;
	// Original cross/joint bilateral filtering
	tm.start();
	for(int c=0;c<int(src.channels());++c)
		apply_crossjoint_bilateral_filter_original(srcsp[c],guidsp[c],dstsp0[c],sigmaS,sigmaR);
		//apply_bilateral_filter_original(srcsp[c],dstsp0[c],sigmaS,sigmaR);
	tm.stop();
	std::cerr<<cv::format("Original BF:     %7.1f [ms]",tm.getTimeMilli())<<std::endl;
	tm.reset();
	// Compressive cross/joint bilateral filtering
	tm.start();
	compressive_bilateral_filter cbf(sigmaS,sigmaR,tol);
	for(int c=0;c<int(src.channels());++c)
		cbf(srcsp[c],dstsp1[c]);
		//cbf(srcsp[c],guidsp[c],dstsp1[c]);
	tm.stop();
	std::cerr<<cv::format("Compressive BF:  %7.1f [ms]",tm.getTimeMilli())<<std::endl;
	tm.reset();
	
	cv::Mat dst0,dst1;
	cv::merge(dstsp0,dst0);
	cv::merge(dstsp1,dst1);
//	clip(dst0); // for debug
//	clip(dst1); // for debug

	double psnr=calc_psnr(dst0,dst1,tone-1.0);
	std::cerr<<cv::format("PSNR:  %f",psnr)<<std::endl;

	if(sw_imshow)
	{
		//cv::imshow("src",src);
		//cv::imshow("guide",guide);
		cv::imshow("dst0",dst0/(tone-1.0));
		cv::imshow("dst1",dst1/(tone-1.0));
		//cv::imshow("error",(dst1-dst0)/(tone-1.0)+0.5);
		cv::waitKey();
	}
	if(sw_imwrite)
	{
		//cv::imwrite("../dst0.png",dst0*(tone-1.0));
		//cv::imwrite("../dst1.png",dst1*(tone-1.0));
		//cv::imwrite("../error.png",(dst1-dst0)+tone/2.0);
	}
	return 0;
}

int main(int argc,char** argv)
{
	// parameters of BF algorithms
	const double sigmaS=2.0;
	const double sigmaR=0.1*(tone-1.0);
	const double tol=0.1; // for compressive BF
	
	if(argc==2)
	{
		//test_bilateral_filter("../lenna-gray.png"); // for debug
		test_bilateral_filter(argv[1],sigmaS,sigmaR,tol);
	}
	else if(argc==3)
	{
		//test_crossjoint_bilateral_filter("../flash0s.png","../flash1s.png",sigmaS,sigmaR,tol); // for debug
		test_crossjoint_bilateral_filter(argv[1],argv[2],sigmaS,sigmaR,tol);
	}
	else
	{
		std::cerr<<"Usage: cbf [InputImagePath] (GuideImagePath)"<<std::endl;
		return 1;
	}
	return 0;
}
