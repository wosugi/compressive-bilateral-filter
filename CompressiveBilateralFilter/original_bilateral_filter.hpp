////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015 Kenjiro Sugimoto
// Released under the MIT license
// http://opensource.org/licenses/mit-license.php
////////////////////////////////////////////////////////////////////////////////
#pragma once
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

/// Original cross/joint bilateral filtering
/// "guide" has to have dynamic range [0,tone).
void apply_crossjoint_bilateral_filter(const cv::Mat_<double>& src,const cv::Mat_<double>& guide,cv::Mat_<double>& dst,double sigmaS,double sigmaR,int tone=256)
{
	assert(src.size()==guide.size());
	assert(src.size()==dst.size());
	
	const int w=src.cols;
	const int h=src.rows;

	// generating spatial kernel
	int r=int(ceil(4.0*sigmaS));
	cv::Mat_<double> kernelS(1+r,1+r);
	for(int v=0;v<=r;++v)
	for(int u=0;u<=r;++u)
		kernelS(v,u)=exp(-0.5*(u*u+v*v)/(sigmaS*sigmaS));
	
	// generating range kernel (discretized for fast computation)
	std::vector<double> kernelR(tone);
	for(int t=0;t<tone;++t)
		kernelR[t]=exp(-0.5*t*t/(sigmaR*sigmaR));
	
	// filtering
	for(int y=0;y<h;++y)
	for(int x=0;x<w;++x)
	{
		double p=src(y,x);
		int t=int(guide(y,x)); // from guide image

		double numer=1.0, denom=p; // (0,0)
		for(int u=1;u<=r;++u) // (u,0)
		{
			double p0=src(y,atW(x-u));
			double p1=src(y,atE(x+u));
			int t0=int(guide(y,atW(x-u))); // from guide image
			int t1=int(guide(y,atE(x+u))); // from guide image
			double wr0=kernelR[abs(t0-t)];
			double wr1=kernelR[abs(t1-t)];
			numer+=kernelS(0,u)*(wr0   +wr1   );
			denom+=kernelS(0,u)*(wr0*p0+wr1*p1);
		}
		for(int v=1;v<=r;++v) // (0,v)
		{
			double p0=src(atN(y-v),x);
			double p1=src(atS(y+v),x);
			int t0=int(guide(atN(y-v),x)); // from guide image
			int t1=int(guide(atS(y+v),x)); // from guide image
			double wr0=kernelR[abs(t0-t)];
			double wr1=kernelR[abs(t1-t)];
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
			int t00=int(guide(atN(y-v),atW(x-u))); // from guide image
			int t01=int(guide(atS(y+v),atW(x-u))); // from guide image
			int t10=int(guide(atN(y-v),atE(x+u))); // from guide image
			int t11=int(guide(atS(y+v),atE(x+u))); // from guide image
			double wr00=kernelR[abs(t00-t)];
			double wr01=kernelR[abs(t01-t)];
			double wr10=kernelR[abs(t10-t)];
			double wr11=kernelR[abs(t11-t)];
			numer+=kernelS(v,u)*(wr00    +wr01    +wr10    +wr11    );
			denom+=kernelS(v,u)*(wr00*p00+wr01*p01+wr10*p10+wr11*p11);
		}
		dst(y,x)=denom/numer;
	}
}
/// Original bilateral filtering
/// "src" has to have dynamic range [0,tone).
void apply_bilateral_filter(const cv::Mat_<double>& src,cv::Mat_<double>& dst,double sigmaS,double sigmaR,int tone=256)
{
	assert(src.size()==dst.size());
	
	const int w=src.cols;
	const int h=src.rows;

	// generating spatial kernel
	int r=int(ceil(4.0*sigmaS));
	cv::Mat_<double> kernelS(1+r,1+r);
	for(int v=0;v<=r;++v)
	for(int u=0;u<=r;++u)
		kernelS(v,u)=exp(-0.5*(u*u+v*v)/(sigmaS*sigmaS));
	
	// generating range kernel (discretized for fast computation)
	std::vector<double> kernelR(tone);
	for(int t=0;t<tone;++t)
		kernelR[t]=exp(-0.5*t*t/(sigmaR*sigmaR));
	
	// filtering
	for(int y=0;y<h;++y)
	for(int x=0;x<w;++x)
	{
		double p=src(y,x);
		int t=int(p);

		double numer=1.0, denom=p; // (0,0)
		for(int u=1;u<=r;++u) // (u,0)
		{
			double p0=src(y,atW(x-u));
			double p1=src(y,atE(x+u));
			int t0=int(p0);
			int t1=int(p1);
			double wr0=kernelR[abs(t0-t)];
			double wr1=kernelR[abs(t1-t)];
			numer+=kernelS(0,u)*(wr0   +wr1   );
			denom+=kernelS(0,u)*(wr0*p0+wr1*p1);
		}
		for(int v=1;v<=r;++v) // (0,v)
		{
			double p0=src(atN(y-v),x);
			double p1=src(atS(y+v),x);
			int t0=int(p0);
			int t1=int(p1);
			double wr0=kernelR[abs(t0-t)];
			double wr1=kernelR[abs(t1-t)];
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
			int t00=int(p00);
			int t01=int(p01);
			int t10=int(p10);
			int t11=int(p11);
			double wr00=kernelR[abs(t00-t)];
			double wr01=kernelR[abs(t01-t)];
			double wr10=kernelR[abs(t10-t)];
			double wr11=kernelR[abs(t11-t)];
			numer+=kernelS(v,u)*(wr00    +wr01    +wr10    +wr11    );
			denom+=kernelS(v,u)*(wr00*p00+wr01*p01+wr10*p10+wr11*p11);
		}
		dst(y,x)=denom/numer;
	}
}
