#pragma
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

/// CB|ABCDE|DC (cv::BORDER_REFLECT_101)
#define atW(x) (std::abs(x))
#define atN(y) (std::abs(y))
#define atE(x) (w-1-std::abs(w-1-(x)))
#define atS(y) (h-1-std::abs(h-1-(y)))

template<int TONE>
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
	std::vector<double> kernelR(TONE);
	const double gamma=1.0/(2.0*sr*sr);
	for(int p=0;p<TONE;++p)
	{
		double t=double(p)/double(TONE-1);
		kernelR[p]=exp(-gamma*t*t);
	}
	
	// filtering
	for(int y=0;y<h;++y)
	for(int x=0;x<w;++x)
	{
		double p=src(y,x);

		double numer=1.0, denom=p; // (0,0)
		for(int u=1;u<=r;++u) // (u,0)
		{
			double p0=src(y,atW(x-u));
			double p1=src(y,atE(x+u));
			double wr0=kernelR[abs(int((TONE-1)*(p0-p)))];
			double wr1=kernelR[abs(int((TONE-1)*(p1-p)))];
			numer+=kernelS(0,u)*(wr0   +wr1   );
			denom+=kernelS(0,u)*(wr0*p0+wr1*p1);
		}
		for(int v=1;v<=r;++v) // (0,v)
		{
			double p0=src(atN(y-v),x);
			double p1=src(atS(y+v),x);
			double wr0=kernelR[abs(int((TONE-1)*(p0-p)))];
			double wr1=kernelR[abs(int((TONE-1)*(p1-p)))];
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
			double wr00=kernelR[abs(int((TONE-1)*(p00-p)))];
			double wr01=kernelR[abs(int((TONE-1)*(p01-p)))];
			double wr10=kernelR[abs(int((TONE-1)*(p10-p)))];
			double wr11=kernelR[abs(int((TONE-1)*(p11-p)))];
			numer+=kernelS(v,u)*(wr00    +wr01    +wr10    +wr11    );
			denom+=kernelS(v,u)*(wr00*p00+wr01*p01+wr10*p10+wr11*p11);
		}
		dst(y,x)=denom/numer;
	}
}
