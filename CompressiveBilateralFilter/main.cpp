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

const int tone=255; // maximum value of 8-bits dynamic range, ie [0,255]

void apply_bilateral_filter_original(const cv::Mat_<double>& src,cv::Mat_<double>& dst,double ss,double sr)
{
	assert(src.size()==dst.size());
	//assert(0.5<=ss && ss<=32.0);
	//assert(0.01<=sr && sr<=0.3);
	
	const int w=src.cols;
	const int h=src.rows;

	// generating spatial kernel
	int r=int(ceil(3.0*ss));
	cv::Mat_<double> kernelS(1+r,1+r);
	for(int v=0;v<=r;++v)
	for(int u=0;u<=r;++u)
		kernelS(v,u)=exp(-(u*u+v*v)/(2.0*ss*ss));
	
	// generating range kernel (discretized for fast computation)
	std::vector<double> kernelR(tone+1);
	const double gamma=1.0/(2.0*sr*sr);
	for(int t=0;t<=tone;++t)
		kernelR[t]=exp(-gamma*t*t/(tone*tone));

	// filtering
	for(int y=0;y<h;++y)
	for(int x=0;x<w;++x)
	{
		double p=src(y,x);

		double numer=1.0;
		double denom=p;
		for(int u=1;u<=r;++u)
		{
			double p0=src(y,atW(x-u));
			double p1=src(y,atE(x+u));
			double wr0=kernelR[abs(int(tone*(p0-p)))];
			double wr1=kernelR[abs(int(tone*(p1-p)))];
			numer+=kernelS(0,u)*(wr0   +wr1   );
			denom+=kernelS(0,u)*(wr0*p0+wr1*p1);
		}
		for(int v=1;v<=r;++v)
		{
			double p0=src(atN(y-v),x);
			double p1=src(atS(y+v),x);
			double wr0=kernelR[abs(int(tone*(p0-p)))];
			double wr1=kernelR[abs(int(tone*(p1-p)))];
			numer+=kernelS(v,0)*(wr0   +wr1   );
			denom+=kernelS(v,0)*(wr0*p0+wr1*p1);
		}
		for(int v=1;v<=r;++v)
		for(int u=1;u<=r;++u)
		{
			double p00=src(atN(y-v),atW(x-u));
			double p01=src(atS(y+v),atW(x-u));
			double p10=src(atN(y-v),atE(x+u));
			double p11=src(atS(y+v),atE(x+u));
			double wr00=kernelR[abs(int(tone*(p00-p)))];
			double wr01=kernelR[abs(int(tone*(p01-p)))];
			double wr10=kernelR[abs(int(tone*(p10-p)))];
			double wr11=kernelR[abs(int(tone*(p11-p)))];
			numer+=kernelS(v,u)*(wr00    +wr01    +wr10    +wr11    );
			denom+=kernelS(v,u)*(wr00*p00+wr01*p01+wr10*p10+wr11*p11);
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

	cv::Mat_<double> image=image0/double(tone); // dynamic range is transformed to [0,1]
	cv::Mat_<double> dst0(image.size());
	cv::Mat_<double> dst1(image.size());

	double ss=2.0;
	double sr=0.1;
	std::cerr<<cv::format("[ss=%f sr=%f]",ss,sr)<<std::endl;

	cv::TickMeter tm;
	
	tm.start();
	apply_bilateral_filter_original(image,dst0,ss,sr);
	tm.stop();
	std::cerr<<cv::format("%7.1f [ms]",tm.getTimeMilli())<<std::endl;
	tm.reset();

	//cv::imwrite(pathO,dst1*tone); // dynamic range is transformed back to [0,tone]
	//cv::imshow("src",image);
	cv::imshow("dst0",dst0);
	cv::waitKey();
	return 0;
}
