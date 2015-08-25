////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015 Kenjiro Sugimoto
// Released under the MIT license
// http://opensource.org/licenses/mit-license.php
////////////////////////////////////////////////////////////////////////////////

//  This code is an implementation of the following two papers. Please cite 
//  BOTH PAPERS in your paper if your research uses this code.
//    1. K. Sugimoto and S. Kamata: "Compressive bilateral filtering", IEEE 
//       Trans. Image Process., vol. 24, no. 11, pp. 3357-3369 (Nov. 2015).
//    2. K. Sugimoto and S. Kamata: "Efficient constant-time Gaussian filtering 
//       with sliding DCT/DST-5 and dual-domain error minimization", ITE Trans. 
//       Media Technol. Appl., vol. 3, no. 1, pp. 12-21 (Jan. 2015).

#pragma once
#define _USE_MATH_DEFINES
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <boost/math/special_functions/erf.hpp>

/// CB|ABCDE|DC (cv::BORDER_REFLECT_101)
#define atW(x) (std::abs(x))
#define atN(y) (std::abs(y))
#define atE(x) (w-1-std::abs(w-1-(x)))
#define atS(y) (h-1-std::abs(h-1-(y)))

//==============================================================================

template<int K>
class constant_time_spatial_gaussian_filter
{
private:
	int r;
	std::vector<double> table,coef1,coefR;

public:
	constant_time_spatial_gaussian_filter(double sigma)
	{
		if(sigma<1.0)
			throw std::invalid_argument("Out of range! (sigma<1.0)");
		
		// estimating the optimal filter window size
		r=estimate_optimal_radius(sigma);
		
		// generating coefficients etc.
		double omega=2.0*M_PI/(r+1+r);
		table=std::vector<double>(K*(1+r));
		coef1=std::vector<double>(K);
		coefR=std::vector<double>(K);
		for(int k=1;k<=K;++k)
		{
			double ak=2.0*exp(-0.5*omega*omega*sigma*sigma*k*k);
			for(int u=0;u<=r;++u)
				table[K*u+k-1]=ak*cos(omega*k*u);
			coef1[k-1]=table[K*1+k-1]*2.0/ak;
			coefR[k-1]=table[K*r+k-1];
		}
	}
	
public:
	int window_size() const
	{
		return r+1+r;
	}
	
private:
	/// a scale-adjusted derivative of the estimated Gaussian spatial kernel error
	class derivative_estimated_gaussian_spatial_kernel_error
	{
	private:
		double s;
		int K;
	public:
		derivative_estimated_gaussian_spatial_kernel_error(double s,int K):s(s),K(K){}
	public:
		double operator()(int r)
		{
			double phi=(2*r+1)/(2.0*s); // spatial domain
			double psi=M_PI*s*(2*K+1)/(2*r+1); // spectral domain
			return phi*exp(-phi*phi)-psi*exp(-psi*psi);
		}
	};
	/// solve df(x)==0 by binary search for integer x=[x1,x2)
	template<class Functor>
	inline int solve_by_bs(Functor df,int x1,int x2) const
	{
		while(1<x2-x1)
		{
			int x=(x1+x2)/2;
			((0.0<df(x))?x2:x1)=x;
		}
		return (abs(df(x1))<=abs(df(x2)))?x1:x2;
	}
	inline int estimate_optimal_radius(double sigma) const
	{
		derivative_estimated_gaussian_spatial_kernel_error df(sigma,K);
		int r=solve_by_bs(df,int(2.0*sigma),int(4.0*sigma));
		return r;
	}
	
	/// this function allows in-place filtering.
	template<int CH>
	inline void filter_x(int w,int h,double* src,double* dst)
	{
		throw std::invalid_argument("Unimplemented channel!");
	}
	template<int CH>
	inline void filter_y(int w,int h,double* src,double* dst)
	{
		throw std::invalid_argument("Unimplemented channel!");
	}

public:
	template<typename T,int CH>
	void filter_x(int w,int h,T* src,T* dst)
	{
		if(w<window_size())
			throw std::invalid_argument("Image width has to be larger than filter window size!");
		filter_x<CH>(w,h,src,dst);
	}
	template<typename T,int CH>
	void filter_y(int w,int h,T* src,T* dst)
	{
		if(h<window_size())
			throw std::invalid_argument("Image height has to be larger than filter window size!");
		filter_y<CH>(w,h,src,dst);
	}
	template<typename T,int CH>
	void filter_xy(int w,int h,T* src,T* dst)
	{
		filter_y<T,CH>(w,h,src,dst);
		filter_x<T,CH>(w,h,dst,dst); // in-place filtering
	}

	/// OpenCV2 interface for easy function call
	void filter_xy(const cv::Mat& src,cv::Mat& dst)
	{
		// checking the format of input/output images
		if(src.size()!=dst.size())
			throw std::invalid_argument("\'src\' and \'dst\' should have the same size!");
		if(src.type()!=dst.type())
			throw std::invalid_argument("\'src\' and \'dst\' should have the same element type!");
		if(src.isSubmatrix() || dst.isSubmatrix())
			throw std::invalid_argument("Subimages are unsupported!");

		switch(src.type())
		{
	//	case CV_32FC1: filter_xy< float,1>(src.cols,src.rows,reinterpret_cast< float*>(src.data),reinterpret_cast< float*>(dst.data)); break;
	//	case CV_32FC4: filter_xy< float,4>(src.cols,src.rows,reinterpret_cast< float*>(src.data),reinterpret_cast< float*>(dst.data)); break;
		case CV_64FC1: filter_xy<double,1>(src.cols,src.rows,reinterpret_cast<double*>(src.data),reinterpret_cast<double*>(dst.data)); break;
		case CV_64FC4: filter_xy<double,4>(src.cols,src.rows,reinterpret_cast<double*>(src.data),reinterpret_cast<double*>(dst.data)); break;
		default: throw std::invalid_argument("Unsupported element type or channel!"); break;
		}
	}
};

//------------------------------------------------------------------------------

template<> template<>
void constant_time_spatial_gaussian_filter<1>::filter_x<1>(int w,int h,double* src,double* dst)
{
	const int K=1,CH=1;
	
	const int r=this->r;
	const std::vector<double> table=this->table;
	const double cf11=coef1[0], cfR1=coefR[0];

	std::vector<double> diff(CH*w); // for in-place filtering
	for(int y=0;y<h;++y)
	{
		double* p=&src[CH*w*y];
		double* q=&dst[CH*w*y];
		
		// preparing initial entries
		double dc0=p[0], a0=p[0]*table[0], b0=p[CH+0]*table[0];
		for(int u=1;u<=r;++u)
		{
			const double sumA0=p[CH*atW(0-u)+0]+p[CH*(0+u)+0], sumB0=p[CH*atW(1-u)+0]+p[CH*(1+u)+0];
			dc0+=sumA0; a0+=sumA0*table[K*u+0]; b0+=sumB0*table[K*u+0];
		}

		// calculating difference values in advance
		for(int x=0;x<w;++x)
		{
			const double* pW=&p[CH*atW(x-r  )];
			const double* pE=&p[CH*atE(x+r+1)];
			diff[CH*x+0]=pE[0]-pW[0];
		}

		double dA0,dB0;
		
		// the first pixel (x=0)
		q[CH*0+0]=dc0+a0; dA0=diff[CH*0+0]; dc0+=dA0;
		
		// the other pixels (x=1,2,...,w-1)
		int x=1;
		while(true) // with 4-length ring buffer
		{
			q[CH*x+0]=dc0+b0; dB0=diff[CH*x+0]; dc0+=dB0; a0+=-cf11*b0+cfR1*(dA0-dB0);
			x++; if(w<=x) break;
			q[CH*x+0]=dc0-a0; dA0=diff[CH*x+0]; dc0+=dA0; b0+=+cf11*a0+cfR1*(dB0-dA0);
			x++; if(w<=x) break;
			q[CH*x+0]=dc0-b0; dB0=diff[CH*x+0]; dc0+=dB0; a0+=-cf11*b0-cfR1*(dA0-dB0);
			x++; if(w<=x) break;
			q[CH*x+0]=dc0+a0; dA0=diff[CH*x+0]; dc0+=dA0; b0+=+cf11*a0-cfR1*(dB0-dA0);
			x++; if(w<=x) break;
		}
	}
}
template<> template<>
void constant_time_spatial_gaussian_filter<2>::filter_x<1>(int w,int h,double* src,double* dst)
{
	const int K=2,CH=1;
	
	const int r=this->r;
	const std::vector<double> table=this->table;
	const double cf11=coef1[0], cfR1=coefR[0];
	const double cf12=coef1[1], cfR2=coefR[1];

	std::vector<double> diff(CH*w); // for in-place filtering
	for(int y=0;y<h;++y)
	{
		double* p=&src[CH*w*y];
		double* q=&dst[CH*w*y];
		
		// preparing initial entries
		double dc0=p[0], a0=p[0]*table[0], b0=p[CH+0]*table[0], aa0=p[0]*table[1], bb0=p[CH+0]*table[1];
		for(int u=1;u<=r;++u)
		{
			const double sumA0=p[CH*atW(0-u)+0]+p[CH*(0+u)+0], sumB0=p[CH*atW(1-u)+0]+p[CH*(1+u)+0];
			dc0+=sumA0; a0+=sumA0*table[K*u+0]; b0+=sumB0*table[K*u+0]; aa0+=sumA0*table[K*u+1]; bb0+=sumB0*table[K*u+1];
		}

		// calculating difference values in advance
		for(int x=0;x<w;++x)
		{
			const double* pW=&p[CH*atW(x-r  )];
			const double* pE=&p[CH*atE(x+r+1)];
			diff[CH*x+0]=pE[0]-pW[0];
		}

		double dA0,dB0,delta0;
		
		// the first pixel (x=0)
		q[CH*0+0]=dc0+a0+aa0; dA0=diff[CH*0+0]; dc0+=dA0;
		
		// the other pixels (x=1,2,...,w-1)
		int x=1;
		while(true) // with 4-length ring buffer
		{
			q[CH*x+0]=dc0+b0+bb0; dB0=diff[CH*x+0]; delta0=dA0-dB0; dc0+=dB0; a0+=-cf11*b0+cfR1*delta0; aa0+=-cf12*bb0+cfR2*delta0;
			x++; if(w<=x) break;
			q[CH*x+0]=dc0-a0-aa0; dA0=diff[CH*x+0]; delta0=dB0-dA0; dc0+=dA0; b0+=+cf11*a0+cfR1*delta0; bb0+=+cf12*aa0+cfR2*delta0;
			x++; if(w<=x) break;
			q[CH*x+0]=dc0-b0-bb0; dB0=diff[CH*x+0]; delta0=dA0-dB0; dc0+=dB0; a0+=-cf11*b0-cfR1*delta0; aa0+=-cf12*bb0-cfR2*delta0;
			x++; if(w<=x) break;
			q[CH*x+0]=dc0+a0+aa0; dA0=diff[CH*x+0]; delta0=dB0-dA0; dc0+=dA0; b0+=+cf11*a0-cfR1*delta0; bb0+=+cf12*aa0-cfR2*delta0;
			x++; if(w<=x) break;
		}
	}
}
template<> template<>
void constant_time_spatial_gaussian_filter<1>::filter_x<4>(int w,int h,double* src,double* dst)
{
	const int K=1,CH=4;
	
	const int r=this->r;
	const std::vector<double> table=this->table;
	const double cf11=coef1[0], cfR1=coefR[0];

	std::vector<double> diff(CH*w); // for in-place filtering
	for(int y=0;y<h;++y)
	{
		double* p=&src[CH*w*y];
		double* q=&dst[CH*w*y];
		
		// preparing initial entries
		double dc0=p[0], a0=p[0]*table[0], b0=p[CH+0]*table[0];
		double dc1=p[1], a1=p[1]*table[0], b1=p[CH+1]*table[0];
		double dc2=p[2], a2=p[2]*table[0], b2=p[CH+2]*table[0];
		double dc3=p[3], a3=p[3]*table[0], b3=p[CH+3]*table[0];
		for(int u=1;u<=r;++u)
		{
			const double sumA0=p[CH*atW(0-u)+0]+p[CH*(0+u)+0], sumB0=p[CH*atW(1-u)+0]+p[CH*(1+u)+0];
			const double sumA1=p[CH*atW(0-u)+1]+p[CH*(0+u)+1], sumB1=p[CH*atW(1-u)+1]+p[CH*(1+u)+1];
			const double sumA2=p[CH*atW(0-u)+2]+p[CH*(0+u)+2], sumB2=p[CH*atW(1-u)+2]+p[CH*(1+u)+2];
			const double sumA3=p[CH*atW(0-u)+3]+p[CH*(0+u)+3], sumB3=p[CH*atW(1-u)+3]+p[CH*(1+u)+3];
			dc0+=sumA0; a0+=sumA0*table[K*u+0]; b0+=sumB0*table[K*u+0];
			dc1+=sumA1; a1+=sumA1*table[K*u+0]; b1+=sumB1*table[K*u+0];
			dc2+=sumA2; a2+=sumA2*table[K*u+0]; b2+=sumB2*table[K*u+0];
			dc3+=sumA3; a3+=sumA3*table[K*u+0]; b3+=sumB3*table[K*u+0];
		}

		// calculating difference values in advance
		for(int x=0;x<w;++x)
		{
			const double* pW=&p[CH*atW(x-r  )];
			const double* pE=&p[CH*atE(x+r+1)];
			diff[CH*x+0]=pE[0]-pW[0];
			diff[CH*x+1]=pE[1]-pW[1];
			diff[CH*x+2]=pE[2]-pW[2];
			diff[CH*x+3]=pE[3]-pW[3];
		}

		double dA0,dB0;
		double dA1,dB1;
		double dA2,dB2;
		double dA3,dB3;
		
		// the first pixel (x=0)
		q[CH*0+0]=dc0+a0; dA0=diff[CH*0+0]; dc0+=dA0;
		q[CH*0+1]=dc1+a1; dA1=diff[CH*0+1]; dc1+=dA1;
		q[CH*0+2]=dc2+a2; dA2=diff[CH*0+2]; dc2+=dA2;
		q[CH*0+3]=dc3+a3; dA3=diff[CH*0+3]; dc3+=dA3;
		
		// the other pixels (x=1,2,...,w-1)
		int x=1;
		while(true) // with 4-length ring buffer
		{
			q[CH*x+0]=dc0+b0; dB0=diff[CH*x+0]; dc0+=dB0; a0+=-cf11*b0+cfR1*(dA0-dB0);
			q[CH*x+1]=dc1+b1; dB1=diff[CH*x+1]; dc1+=dB1; a1+=-cf11*b1+cfR1*(dA1-dB1);
			q[CH*x+2]=dc2+b2; dB2=diff[CH*x+2]; dc2+=dB2; a2+=-cf11*b2+cfR1*(dA2-dB2);
			q[CH*x+3]=dc3+b3; dB3=diff[CH*x+3]; dc3+=dB3; a3+=-cf11*b3+cfR1*(dA3-dB3);
			x++; if(w<=x) break;
			q[CH*x+0]=dc0-a0; dA0=diff[CH*x+0]; dc0+=dA0; b0+=+cf11*a0+cfR1*(dB0-dA0);
			q[CH*x+1]=dc1-a1; dA1=diff[CH*x+1]; dc1+=dA1; b1+=+cf11*a1+cfR1*(dB1-dA1);
			q[CH*x+2]=dc2-a2; dA2=diff[CH*x+2]; dc2+=dA2; b2+=+cf11*a2+cfR1*(dB2-dA2);
			q[CH*x+3]=dc3-a3; dA3=diff[CH*x+3]; dc3+=dA3; b3+=+cf11*a3+cfR1*(dB3-dA3);
			x++; if(w<=x) break;
			q[CH*x+0]=dc0-b0; dB0=diff[CH*x+0]; dc0+=dB0; a0+=-cf11*b0-cfR1*(dA0-dB0);
			q[CH*x+1]=dc1-b1; dB1=diff[CH*x+1]; dc1+=dB1; a1+=-cf11*b1-cfR1*(dA1-dB1);
			q[CH*x+2]=dc2-b2; dB2=diff[CH*x+2]; dc2+=dB2; a2+=-cf11*b2-cfR1*(dA2-dB2);
			q[CH*x+3]=dc3-b3; dB3=diff[CH*x+3]; dc3+=dB3; a3+=-cf11*b3-cfR1*(dA3-dB3);
			x++; if(w<=x) break;
			q[CH*x+0]=dc0+a0; dA0=diff[CH*x+0]; dc0+=dA0; b0+=+cf11*a0-cfR1*(dB0-dA0);
			q[CH*x+1]=dc1+a1; dA1=diff[CH*x+1]; dc1+=dA1; b1+=+cf11*a1-cfR1*(dB1-dA1);
			q[CH*x+2]=dc2+a2; dA2=diff[CH*x+2]; dc2+=dA2; b2+=+cf11*a2-cfR1*(dB2-dA2);
			q[CH*x+3]=dc3+a3; dA3=diff[CH*x+3]; dc3+=dA3; b3+=+cf11*a3-cfR1*(dB3-dA3);
			x++; if(w<=x) break;
		}
	}
}
template<> template<>
void constant_time_spatial_gaussian_filter<2>::filter_x<4>(int w,int h,double* src,double* dst)
{
	const int K=2,CH=4;
	
	const int r=this->r;
	const std::vector<double> table=this->table;
	const double cf11=coef1[0], cfR1=coefR[0];
	const double cf12=coef1[1], cfR2=coefR[1];

	std::vector<double> diff(CH*w); // for in-place filtering
	for(int y=0;y<h;++y)
	{
		double* p=&src[CH*w*y];
		double* q=&dst[CH*w*y];
		
		// preparing initial entries
		double dc0=p[0], a0=p[0]*table[0], b0=p[CH+0]*table[0], aa0=p[0]*table[1], bb0=p[CH+0]*table[1];
		double dc1=p[1], a1=p[1]*table[0], b1=p[CH+1]*table[0], aa1=p[1]*table[1], bb1=p[CH+1]*table[1];
		double dc2=p[2], a2=p[2]*table[0], b2=p[CH+2]*table[0], aa2=p[2]*table[1], bb2=p[CH+2]*table[1];
		double dc3=p[3], a3=p[3]*table[0], b3=p[CH+3]*table[0], aa3=p[3]*table[1], bb3=p[CH+3]*table[1];
		for(int u=1;u<=r;++u)
		{
			const double sumA0=p[CH*atW(0-u)+0]+p[CH*(0+u)+0], sumB0=p[CH*atW(1-u)+0]+p[CH*(1+u)+0];
			const double sumA1=p[CH*atW(0-u)+1]+p[CH*(0+u)+1], sumB1=p[CH*atW(1-u)+1]+p[CH*(1+u)+1];
			const double sumA2=p[CH*atW(0-u)+2]+p[CH*(0+u)+2], sumB2=p[CH*atW(1-u)+2]+p[CH*(1+u)+2];
			const double sumA3=p[CH*atW(0-u)+3]+p[CH*(0+u)+3], sumB3=p[CH*atW(1-u)+3]+p[CH*(1+u)+3];
			dc0+=sumA0; a0+=sumA0*table[K*u+0]; b0+=sumB0*table[K*u+0]; aa0+=sumA0*table[K*u+1]; bb0+=sumB0*table[K*u+1];
			dc1+=sumA1; a1+=sumA1*table[K*u+0]; b1+=sumB1*table[K*u+0]; aa1+=sumA1*table[K*u+1]; bb1+=sumB1*table[K*u+1];
			dc2+=sumA2; a2+=sumA2*table[K*u+0]; b2+=sumB2*table[K*u+0]; aa2+=sumA2*table[K*u+1]; bb2+=sumB2*table[K*u+1];
			dc3+=sumA3; a3+=sumA3*table[K*u+0]; b3+=sumB3*table[K*u+0]; aa3+=sumA3*table[K*u+1]; bb3+=sumB3*table[K*u+1];
		}

		// calculating difference values in advance
		for(int x=0;x<w;++x)
		{
			const double* pW=&p[CH*atW(x-r  )];
			const double* pE=&p[CH*atE(x+r+1)];
			diff[CH*x+0]=pE[0]-pW[0];
			diff[CH*x+1]=pE[1]-pW[1];
			diff[CH*x+2]=pE[2]-pW[2];
			diff[CH*x+3]=pE[3]-pW[3];
		}

		double dA0,dB0,delta0;
		double dA1,dB1,delta1;
		double dA2,dB2,delta2;
		double dA3,dB3,delta3;
		
		// the first pixel (x=0)
		q[CH*0+0]=dc0+a0+aa0; dA0=diff[CH*0+0]; dc0+=dA0;
		q[CH*0+1]=dc1+a1+aa1; dA1=diff[CH*0+1]; dc1+=dA1;
		q[CH*0+2]=dc2+a2+aa2; dA2=diff[CH*0+2]; dc2+=dA2;
		q[CH*0+3]=dc3+a3+aa3; dA3=diff[CH*0+3]; dc3+=dA3;
		
		// the other pixels (x=1,2,...,w-1)
		int x=1;
		while(true) // with 4-length ring buffer
		{
			q[CH*x+0]=dc0+b0+bb0; dB0=diff[CH*x+0]; delta0=dA0-dB0; dc0+=dB0; a0+=-cf11*b0+cfR1*delta0; aa0+=-cf12*bb0+cfR2*delta0;
			q[CH*x+1]=dc1+b1+bb1; dB1=diff[CH*x+1]; delta1=dA1-dB1; dc1+=dB1; a1+=-cf11*b1+cfR1*delta1; aa1+=-cf12*bb1+cfR2*delta1;
			q[CH*x+2]=dc2+b2+bb2; dB2=diff[CH*x+2]; delta2=dA2-dB2; dc2+=dB2; a2+=-cf11*b2+cfR1*delta2; aa2+=-cf12*bb2+cfR2*delta2;
			q[CH*x+3]=dc3+b3+bb3; dB3=diff[CH*x+3]; delta3=dA3-dB3; dc3+=dB3; a3+=-cf11*b3+cfR1*delta3; aa3+=-cf12*bb3+cfR2*delta3;
			x++; if(w<=x) break;
			q[CH*x+0]=dc0-a0-aa0; dA0=diff[CH*x+0]; delta0=dB0-dA0; dc0+=dA0; b0+=+cf11*a0+cfR1*delta0; bb0+=+cf12*aa0+cfR2*delta0;
			q[CH*x+1]=dc1-a1-aa1; dA1=diff[CH*x+1]; delta1=dB1-dA1; dc1+=dA1; b1+=+cf11*a1+cfR1*delta1; bb1+=+cf12*aa1+cfR2*delta1;
			q[CH*x+2]=dc2-a2-aa2; dA2=diff[CH*x+2]; delta2=dB2-dA2; dc2+=dA2; b2+=+cf11*a2+cfR1*delta2; bb2+=+cf12*aa2+cfR2*delta2;
			q[CH*x+3]=dc3-a3-aa3; dA3=diff[CH*x+3]; delta3=dB3-dA3; dc3+=dA3; b3+=+cf11*a3+cfR1*delta3; bb3+=+cf12*aa3+cfR2*delta3;
			x++; if(w<=x) break;
			q[CH*x+0]=dc0-b0-bb0; dB0=diff[CH*x+0]; delta0=dA0-dB0; dc0+=dB0; a0+=-cf11*b0-cfR1*delta0; aa0+=-cf12*bb0-cfR2*delta0;
			q[CH*x+1]=dc1-b1-bb1; dB1=diff[CH*x+1]; delta1=dA1-dB1; dc1+=dB1; a1+=-cf11*b1-cfR1*delta1; aa1+=-cf12*bb1-cfR2*delta1;
			q[CH*x+2]=dc2-b2-bb2; dB2=diff[CH*x+2]; delta2=dA2-dB2; dc2+=dB2; a2+=-cf11*b2-cfR1*delta2; aa2+=-cf12*bb2-cfR2*delta2;
			q[CH*x+3]=dc3-b3-bb3; dB3=diff[CH*x+3]; delta3=dA3-dB3; dc3+=dB3; a3+=-cf11*b3-cfR1*delta3; aa3+=-cf12*bb3-cfR2*delta3;
			x++; if(w<=x) break;
			q[CH*x+0]=dc0+a0+aa0; dA0=diff[CH*x+0]; delta0=dB0-dA0; dc0+=dA0; b0+=+cf11*a0-cfR1*delta0; bb0+=+cf12*aa0-cfR2*delta0;
			q[CH*x+1]=dc1+a1+aa1; dA1=diff[CH*x+1]; delta1=dB1-dA1; dc1+=dA1; b1+=+cf11*a1-cfR1*delta1; bb1+=+cf12*aa1-cfR2*delta1;
			q[CH*x+2]=dc2+a2+aa2; dA2=diff[CH*x+2]; delta2=dB2-dA2; dc2+=dA2; b2+=+cf11*a2-cfR1*delta2; bb2+=+cf12*aa2-cfR2*delta2;
			q[CH*x+3]=dc3+a3+aa3; dA3=diff[CH*x+3]; delta3=dB3-dA3; dc3+=dA3; b3+=+cf11*a3-cfR1*delta3; bb3+=+cf12*aa3-cfR2*delta3;
			x++; if(w<=x) break;
		}
	}
}

//------------------------------------------------------------------------------

template<> template<int CH>
void constant_time_spatial_gaussian_filter<1>::filter_y(int w,int h,double* src,double* dst)
{
	const int K=1;
	
	const int r=this->r;
	const std::vector<double> table=this->table;
	const double cf11=coef1[0], cfR1=coefR[0];

	std::vector<double> workspace(CH*w*(2*K+2)); // work space to keep raster scanning

	// preparing initial entries
	for(int cx=0;cx<CH*w;++cx)
	{
		double* ws=&workspace[cx*(2*K+2)];
		ws[0]=src[cx];
		ws[2]=src[cx+CH*w]*table[0]; ws[3]=src[cx]*table[0];
	}
	for(int v=1;v<=r;++v)
	{
		for(int cx=0;cx<CH*w;++cx)
		{
			const double sum0=src[cx+CH*w*atN(0-v)]+src[cx+CH*w*(0+v)];
			const double sum1=src[cx+CH*w*atN(1-v)]+src[cx+CH*w*(1+v)];
			double* ws=&workspace[cx*(2*K+2)];
			ws[0]+=sum0;
			ws[2]+=sum1*table[K*v+0]; ws[3]+=sum0*table[K*v+0];
		}
	}

	double *q,*p0N,*p1S;
	double diff,delta;
	
	// the first line (y=0)
	{
		q=&dst[CH*w*0];
		p0N=&src[CH*w*atN(0-r  )];
		p1S=&src[CH*w*atS(0+r+1)];
		for(int cx=0;cx<CH*w;++cx)
		{
			double* ws=&workspace[cx*(2*K+2)];
			q[cx]=ws[0]+ws[3];
			diff=p1S[cx]-p0N[cx];
			ws[0]+=diff;
			ws[1]=diff;
		}
	}
	// the other lines (y=1,2,...,h-1)
	int y=1;
	while(true) // with 2-length ring buffers
	{
		q=&dst[CH*w*y];
		p0N=&src[CH*w*atN(y-r  )];
		p1S=&src[CH*w*atS(y+r+1)];
		for(int cx=0;cx<CH*w;++cx)
		{
			double* ws=&workspace[cx*(2*K+2)];
			q[cx]=ws[0]+ws[2];

			diff=p1S[cx]-p0N[cx];
			delta=diff-ws[1];
			ws[0]+=diff;
			ws[3]=cfR1*delta+cf11*ws[2]-ws[3];
			ws[1]=diff;
		}
		y++; if(h<=y) break;
		
		q=&dst[CH*w*y];
		p0N=&src[CH*w*atN(y-r  )];
		p1S=&src[CH*w*atS(y+r+1)];
		for(int cx=0;cx<CH*w;++cx)
		{
			double* ws=&workspace[cx*(2*K+2)];
			q[cx]=ws[0]+ws[3];

			diff=p1S[cx]-p0N[cx];
			delta=diff-ws[1];
			ws[0]+=diff;
			ws[2]=cfR1*delta+cf11*ws[3]-ws[2];
			ws[1]=diff;
		}
		y++; if(h<=y) break;
	}
}
template<> template<int CH>
void constant_time_spatial_gaussian_filter<2>::filter_y(int w,int h,double* src,double* dst)
{
	const int K=2;
	
	const int r=this->r;
	const std::vector<double> table=this->table;
	const double cf11=coef1[0], cfR1=coefR[0];
	const double cf12=coef1[1], cfR2=coefR[1];

	std::vector<double> workspace(CH*w*(2*K+2),0.0); // work space to keep raster scanning

	// preparing initial entries
	for(int cx=0;cx<CH*w;++cx)
	{
		double* ws=&workspace[cx*(2*K+2)];
		ws[0]=src[cx];
		ws[2]=src[cx+CH*w]*table[0]; ws[3]=src[cx]*table[0];
		ws[4]=src[cx+CH*w]*table[1]; ws[5]=src[cx]*table[1];
	}
	for(int v=1;v<=r;++v)
	{
		for(int cx=0;cx<CH*w;++cx)
		{
			const double sum0=src[cx+CH*w*atN(0-v)]+src[cx+CH*w*(0+v)];
			const double sum1=src[cx+CH*w*atN(1-v)]+src[cx+CH*w*(1+v)];
			double* ws=&workspace[cx*(2*K+2)];
			ws[0]+=sum0;
			ws[2]+=sum1*table[K*v+0]; ws[3]+=sum0*table[K*v+0];
			ws[4]+=sum1*table[K*v+1]; ws[5]+=sum0*table[K*v+1];
		}
	}

	double *q,*p0N,*p1S;
	double diff,delta;
	
	// the first line (y=0)
	{
		q=&dst[CH*w*0];
		p0N=&src[CH*w*atN(0-r  )];
		p1S=&src[CH*w*atS(0+r+1)];
		for(int cx=0;cx<CH*w;++cx)
		{
			double* ws=&workspace[cx*(2*K+2)];
			q[cx]=ws[0]+ws[3]+ws[5];
			diff=p1S[cx]-p0N[cx];
			ws[0]+=diff;
			ws[1]=diff;
		}
	}
	// the other lines (y=1,2,...,h-1)
	int y=1;
	while(true) // with 2-length ring buffers
	{
		q=&dst[CH*w*y];
		p0N=&src[CH*w*atN(y-r  )];
		p1S=&src[CH*w*atS(y+r+1)];
		for(int cx=0;cx<CH*w;++cx)
		{
			double* ws=&workspace[cx*(2*K+2)];
			q[cx]=ws[0]+ws[2]+ws[4];

			diff=p1S[cx]-p0N[cx];
			delta=diff-ws[1];
			ws[0]+=diff;
			ws[1]=diff;
			ws[3]=-ws[3]+cf11*ws[2]+cfR1*delta;
			ws[5]=-ws[5]+cf12*ws[4]+cfR2*delta;
		}
		y++; if(h<=y) break;
		
		q=&dst[CH*w*y];
		p0N=&src[CH*w*atN(y-r  )];
		p1S=&src[CH*w*atS(y+r+1)];
		for(int cx=0;cx<CH*w;++cx)
		{
			double* ws=&workspace[cx*(2*K+2)];
			q[cx]=ws[0]+ws[3]+ws[5];
			
			diff=p1S[cx]-p0N[cx];
			delta=diff-ws[1];
			ws[0]+=diff;
			ws[1]=diff;
			ws[2]=-ws[2]+cf11*ws[3]+cfR1*delta;
			ws[4]=-ws[4]+cf12*ws[5]+cfR2*delta;
		}
		y++; if(h<=y) break;
	}
}

//==============================================================================

class compressive_bilateral_filter
{
private:
	// this parameter will provide sufficient accuracy.
	constant_time_spatial_gaussian_filter<1> gaussian;

	int K;
	double T;
	std::vector<double> sqrta;

public:
	compressive_bilateral_filter(double sigmaS,double sigmaR,double tau):gaussian(sigmaS)
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
				compsI(y,x)=cv::Vec4d(cp*src(y,x),sp*src(y,x),cp,sp);
			}
			gaussian.filter_xy(compsI,compsO);
		
			// decompressing k-th components
			for(int y=0;y<src.rows;++y)
			for(int x=0;x<src.cols;++x)
			{
				int p=int(src(y,x)*(tone-1));
				double cp=tblC[p];
				double sp=tblS[p];
				const cv::Vec4d& values=compsO(y,x);
				numer(y,x)+=cp*values[0]+sp*values[1];
				denom(y,x)+=cp*values[2]+sp*values[3];
			}
		}
		dst=numer/denom;
	}
};

//==============================================================================
