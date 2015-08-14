#pragma once
#define _USE_MATH_DEFINES
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <boost/math/special_functions/erf.hpp>

// a scale-adjusted derivative of the estimated kernel error E(T)
class derivative_estimated_kernel_error
{
private:
	double s,kappa;
public:
	derivative_estimated_kernel_error(double s,int K):s(s),kappa(M_PI*(2*K+1)){}
public:
	double operator()(double T)
	{
		double a=kappa*s/T;
		double b=(T-1.0)/s;
		return a*a*exp(-a*a)-kappa*exp(-b*b);
	}
};

// solve df(x)==0 by binary search
template<class Functor>
inline double solve_by_bs(Functor df,double x1,double x2,int loop=10)
{
	for(int i=0;i<loop;++i)
	{
		double x=(x1+x2)/2.0;
		double dy=df(x);
		if(0.0<=dy)
			x2=x;
		else
			x1=x;
	}
	return (x1+x2)/2.0;
}

// s has to be normalized as R=1
inline std::pair<int,double> optimize_K_and_T(double s,double tau)
{
	const double xi=boost::math::erfc_inv(tau*tau);
	int K=static_cast<int>(std::ceil(xi*xi/(2.0*M_PI)+xi/(2.0*M_PI*s)-0.5));

	// It is better to slightly extend the original search domain D
	// because it might uncover the minimum of E(T) due to approximate error.
	const double magicnum=0.03;
	double t1=s*xi+1.0;
	double t2=M_PI*s*(2*K+1)/xi+magicnum;
	double T=solve_by_bs(derivative_estimated_kernel_error(s,K),t1,t2);
	return std::make_pair(K,T);
}

// dynamic range has to be [0,1].
template<int TONE>
void apply_bilateral_filter_compressive(const cv::Mat_<double>& src,cv::Mat_<double>& dst,double ss,double sr,double tau)
{
	// computing parameters
	std::pair<int,double> params=optimize_K_and_T(sr,tau);
	int K=params.first;
	double T=params.second;
	
	// lookup tables (discretized for fast computation)
	std::vector<double> tblC(TONE);
	std::vector<double> tblS(TONE);
	cv::Mat_<cv::Vec4d> comps(src.size()); // component images
	
	// DC component
	cv::Mat_<double> denom(src.size(),1.0);
	cv::Mat_<double> numer(src.size());
	cv::GaussianBlur(src,numer,cv::Size(),ss,ss);
	
	// AC components
	const double omega=2.0*M_PI/T;
	for(int k=1;k<=K;++k)
	{
		// preparing look-up tables
		const double omegak=omega*k;
		double sqrtak=M_SQRT2*exp(-omegak*omegak*sr*sr/4.0);
		for(int p=0;p<TONE;++p)
		{
			double theta=omegak*p/double(TONE-1);
			tblC[p]=sqrtak*cos(theta);
			tblS[p]=sqrtak*sin(theta);
		}

		// generating k-th component images
		for(int y=0;y<src.rows;++y)
		for(int x=0;x<src.cols;++x)
		{
			int p=int(src(y,x)*(TONE-1));
			double cp=tblC[p];
			double sp=tblS[p];
			comps(y,x)=cv::Vec4d(cp*src(y,x),sp*src(y,x),cp,sp);
		}
		cv::GaussianBlur(comps,comps,cv::Size(),ss,ss);
		
		// decompressing k-th components
		for(int y=0;y<src.rows;++y)
		for(int x=0;x<src.cols;++x)
		{
			int p=int(src(y,x)*(TONE-1));
			double cp=tblC[p];
			double sp=tblS[p];
			const cv::Vec4d& values=comps(y,x);
			numer(y,x)+=cp*values[0]+sp*values[1];
			denom(y,x)+=cp*values[2]+sp*values[3];
		}
	}
	dst=numer/denom;
}
