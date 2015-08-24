////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015 Kenjiro Sugimoto
// Released under the MIT license
// http://opensource.org/licenses/mit-license.php
////////////////////////////////////////////////////////////////////////////////
#pragma once
#define _USE_MATH_DEFINES
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <boost/math/special_functions/erf.hpp>
#include "constant_time_gaussian_filter.hpp"

//==================================================================================================

/// This class is an implementation of the following paper:
///   - K. Sugimoto and S. Kamata: "Compressive bilateral filtering",
///     IEEE Trans. Image Process., vol. 24, no. 11, pp. 3357-3369 (Nov. 2015).
/// This paper needs to be cited in your paper if you use this code.
class compressive_bilateral_filter
{
private:
	int K;
	double T;
	double _sigmaS;
	std::vector<double> sqrta;
//	constant_time_spatial_gaussian_filter<1> gaussian;
	constant_time_spatial_gaussian_filter<2> gaussian;

public:
	compressive_bilateral_filter(double sigmaS,double sigmaR,double tau):gaussian(sigmaS),_sigmaS(sigmaS)
	{
		// estimating an optimal K
		double xi=boost::math::erfc_inv(tau*tau);
		K=static_cast<int>(std::ceil(xi*xi/(2.0*M_PI)+xi/(2.0*M_PI*sigmaR)-0.5));
		
		// estimating an optimal T
		// It is better to slightly extend the original search domain D
		// because it might uncover the minimum of E(T) due to approximate error.
		const double magicnum=0.03;
		double t1=sigmaR*xi+1.0;
		double t2=M_PI*sigmaR*(2*K+1)/xi+magicnum;
		derivative_estimated_gaussian_range_kernel_error df(sigmaR,K);
		T=solve_by_bs(df,t1,t2);

		// precomputing the square root of spectrum
		double omega=2.0*M_PI/T;
		sqrta=std::vector<double>(K);
		for(int k=1;k<=K;++k)
			sqrta[k-1]=M_SQRT2*exp(-0.25*omega*omega*sigmaR*sigmaR*k*k);
	}

private:
	/// a scale-adjusted derivative of the estimated Gaussian range kernel error
	class derivative_estimated_gaussian_range_kernel_error
	{
	private:
		double sigma,kappa;
	public:
		derivative_estimated_gaussian_range_kernel_error(double sigma,int K):sigma(sigma),kappa(M_PI*(2*K+1)){}
	public:
		double operator()(double T)
		{
			double phi=(T-1.0)/sigma;
			double psi=kappa*sigma/T;
			return kappa*exp(-phi*phi)-psi*psi*exp(-psi*psi);
		}
	};
	/// solve df(x)==0 by binary search
	template<class Functor>
	inline double solve_by_bs(Functor df,double x1,double x2,int loop=10)
	{
		for(int i=0;i<loop;++i)
		{
			double x=(x1+x2)/2.0;
			((0.0<=df(x))?x2:x1)=x;
		}
		return (x1+x2)/2.0;
	}

public:
	/// dynamic range has to be [0,1].
	void operator()(const cv::Mat_<double>& src,cv::Mat_<double>& dst,int tone=256)
	{
		// lookup tables (discretized for fast computation)
		std::vector<double> tblC(tone);
		std::vector<double> tblS(tone);
		cv::Mat_<cv::Vec4d> comps(src.size()); // component images
	
		// DC component
		cv::Mat_<double> denom(src.size(),1.0);
		cv::Mat_<double> numer(src.size());
		cv::GaussianBlur(src,numer,cv::Size(),_sigmaS,_sigmaS,cv::BORDER_REFLECT_101); // temporal implementation

		// AC components
		double omega=2.0*M_PI/T;
		for(int k=1;k<=K;++k)
		{
			// preparing look-up tables
			const double omegak=omega*k;
			for(int p=0;p<tone;++p)
			{
				double theta=omegak*p/double(tone-1);
				tblC[p]=sqrta[k-1]*cos(theta);
				tblS[p]=sqrta[k-1]*sin(theta);
			}

			// generating k-th component images
			for(int y=0;y<src.rows;++y)
			for(int x=0;x<src.cols;++x)
			{
				int p=int(src(y,x)*(tone-1));
				double cp=tblC[p];
				double sp=tblS[p];
				comps(y,x)=cv::Vec4d(cp*src(y,x),sp*src(y,x),cp,sp);
			}
			gaussian.filter(comps,comps);
		
			// decompressing k-th components
			for(int y=0;y<src.rows;++y)
			for(int x=0;x<src.cols;++x)
			{
				int p=int(src(y,x)*(tone-1));
				double cp=tblC[p];
				double sp=tblS[p];
				const cv::Vec4d& values=comps(y,x);
				numer(y,x)+=cp*values[0]+sp*values[1];
				denom(y,x)+=cp*values[2]+sp*values[3];
			}
		}
		dst=numer/denom;
	}
};

//==================================================================================================
