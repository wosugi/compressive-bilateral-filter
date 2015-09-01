////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015 Kenjiro Sugimoto
// Released under the MIT license
// http://opensource.org/licenses/mit-license.php
////////////////////////////////////////////////////////////////////////////////

// This code implements the algorithm of the following paper. Please cite it in 
// your paper if your research uses this code.
//   + K. Sugimoto and S. Kamata: "Compressive bilateral filtering", IEEE Trans.
//     Image Process., vol. 24, no. 11, pp. 3357-3369 (Nov. 2015).

#pragma once
#define _USE_MATH_DEFINES
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#ifdef USE_OPENCV2
#include <opencv2/opencv.hpp>
#endif
#ifdef USE_BOOST
#include <boost/math/special_functions/erf.hpp> // for erfc_inv() only
#endif
#include "o1_spatial_gaussian_filter.hpp"

//==============================================================================

class compressive_bilateral_filter
{
private:
	int tone;

	// this parameter will provide sufficient accuracy.
	o1_spatial_gaussian_filter<1> gaussian;
	int K; // number of basis range kernels
	double T; // period length of periodic range kernel
	std::vector<double> sqrta;

public:
	compressive_bilateral_filter(double sigmaS,double sigmaR,double tol=0.10,int tone=256):tone(tone),gaussian(sigmaS)
	{
#ifdef USE_BOOST
		double xi=boost::math::erfc_inv(tol*tol);
#else
		// hard-coding for boost-less running
		double xi;
		     if(tol==0.05) xi=2.1378252338818511;
		else if(tol==0.10) xi=1.8213863677184496;
		else if(tol==0.20) xi=1.4522197815622468;
		else
			throw std::invalid_argument("Unsupported tolerance! ({0.05,0.10,0.20} only or use boost)");
#endif
		// estimating an optimal K
		double s=sigmaR/(tone-1.0); // normalized as dynamic range [0,1]
		K=static_cast<int>(std::ceil(xi*xi/(2.0*M_PI)+xi/(2.0*M_PI*s)-0.5));
		
		// estimating an optimal T
		// It is better to slightly extend the original search domain D
		// because it might uncover the minimum of E(T) due to approximate error.
		const double MAGICNUM=0.03;
		double t1=s*xi+1.0;
		double t2=M_PI*s*(2*K+1)/xi+MAGICNUM;
		derivative_estimated_gaussian_range_kernel_error df(s,K);
		T=(tone-1.0)*solve_by_bs(df,t1,t2);

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
		derivative_estimated_gaussian_range_kernel_error(double sigma,int K)
			:sigma(sigma),kappa(M_PI*(2*K+1)){}
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
#ifdef USE_OPENCV2
	/// dynamic range has to be [0,1].
	void operator()(const cv::Mat_<double>& src,cv::Mat_<double>& dst)
	{
		// lookup tables (discretized for fast computation)
		std::vector<double> tblC(tone);
		std::vector<double> tblS(tone);
		// component images
		cv::Mat_<cv::Vec4d> compsI(src.size());
		cv::Mat_<cv::Vec4d> compsO(src.size());
		
		// DC component
		const int winsz=gaussian.window_size();
		cv::Mat_<double> denom(src.size(),winsz*winsz);
		cv::Mat_<double> numer(src.size());
		gaussian.filter_xy(src,numer);

		// AC components
		double omega=2.0*M_PI/T;
		for(int k=1;k<=K;++k)
		{
			// preparing look-up tables
			double omegak=omega*k;
			for(int t=0;t<tone;++t)
			{
				double theta=omegak*t;
				tblC[t]=sqrta[k-1]*cos(theta);
				tblS[t]=sqrta[k-1]*sin(theta);
			}

			// generating k-th component images
			for(int y=0;y<src.rows;++y)
			for(int x=0;x<src.cols;++x)
			{
				int p=int(src(y,x));
				double cp=tblC[p];
				double sp=tblS[p];
				compsI(y,x)=cv::Vec4d(cp*src(y,x),sp*src(y,x),cp,sp);
			}
			gaussian.filter_xy(compsI,compsO);
		
			// decompressing k-th components
			for(int y=0;y<src.rows;++y)
			for(int x=0;x<src.cols;++x)
			{
				int p=int(src(y,x));
				double cp=tblC[p];
				double sp=tblS[p];
				const cv::Vec4d& values=compsO(y,x);
				numer(y,x)+=cp*values[0]+sp*values[1];
				denom(y,x)+=cp*values[2]+sp*values[3];
			}
		}
		dst=numer/denom;
	}
#endif
};

//==============================================================================
