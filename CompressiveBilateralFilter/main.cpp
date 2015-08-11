#define _USE_MATH_DEFINES
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

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

/// CB|ABCDE|DC (cv::BORDER_REFLECT_101)
#define atW(x) (std::abs(x))
#define atN(y) (std::abs(y))
#define atE(x) (w-1-std::abs(w-1-(x)))
#define atS(y) (h-1-std::abs(h-1-(y)))

void apply_bilateral_filter_original(const cv::Mat_<double>& src,cv::Mat_<double>& dst,double ss,double sr)
{
	assert(src.size()==dst.size());
	//assert(0.5<=ss && ss<=32.0);
	//assert(0.01<=sr && sr<=0.3);
	
	const int w=src.cols;
	const int h=src.rows;

	// generating spatial kernel
	int r=int(ceil(3.0*ss));
	cv::Mat_<double> kernel(1+r,1+r);
	double eta=0.0;
	for(int v=0;v<=r;++v)
	for(int u=0;u<=r;++u)
	{
		double weight=exp(-(u*u+v*v)/(2.0*ss*ss));
		if(u==0) weight*=0.5;
		if(v==0) weight*=0.5;
		eta+=4.0*weight;
		kernel(v,u)=weight;
	}
	kernel/=eta; // normalization

	// filtering
	const double gamma=1.0/(2.0*sr*sr);
	for(int y=0;y<h;++y)
	for(int x=0;x<w;++x)
	{
		const double p=src(y,x);
		double numer=0.0;
		double denom=0.0;
		for(int v=0;v<=r;++v)
		for(int u=0;u<=r;++u)
		{
			const double p00=src(atN(y-v),atW(x-u));
			const double p01=src(atS(y+v),atW(x-u));
			const double p10=src(atN(y-v),atE(x+u));
			const double p11=src(atS(y+v),atE(x+u));
			const double wr00=exp(-gamma*(p00-p)*(p00-p));
			const double wr01=exp(-gamma*(p01-p)*(p01-p));
			const double wr10=exp(-gamma*(p10-p)*(p10-p));
			const double wr11=exp(-gamma*(p11-p)*(p11-p));

			numer+=kernel(v,u)*(wr00+wr01+wr10+wr11);
			denom+=kernel(v,u)*(wr00*p00+wr01*p01+wr10*p10+wr11*p11);
		}
		dst(y,x)=denom/numer;
	}
}

int main(int argc,char** argv)
{
	if(argc!=3)
	{
		std::cerr<<"Usage: cbf [ImageInputPath] [ImageOutputPath]"<<std::endl;
		return 1;
	}
	const std::string& pathI(argv[1]);
	const std::string& pathO(argv[2]);

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

	cv::Mat_<double> image=image0/255.0;

	cv::Mat_<double> dst1(image.size());

	double ss=2.0;
	double sr=0.1;

	cv::TickMeter tm;
	tm.reset();
	tm.start();
	apply_bilateral_filter_original(image,dst1,ss,sr);
	tm.stop();

	std::cerr<<cv::format("ss=%f sr=%f :  ",ss,sr)<<std::endl;
	std::cerr<<cv::format("%7.1f [ms]",tm.getTimeMilli())<<std::endl;
	std::cerr<<std::endl;

	cv::imwrite(pathO,dst1*255.0);
	//cv::imshow("src",image);
	//cv::imshow("dst1",dst1);
	//cv::waitKey();
	return 0;
}
