////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015 Kenjiro Sugimoto
// Released under the MIT license
// http://opensource.org/licenses/mit-license.php
////////////////////////////////////////////////////////////////////////////////
#pragma once
#define _USE_MATH_DEFINES
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <opencv2/opencv.hpp>

//==================================================================================================

/// CB|ABCDE|DC (cv::BORDER_REFLECT_101)
#define atW(x) (std::abs(x))
#define atN(y) (std::abs(y))
#define atE(x) (w-1-std::abs(w-1-(x)))
#define atS(y) (h-1-std::abs(h-1-(y)))

static inline double phase(int r)
{
	return 2.0*M_PI/(r+1+r); // the unit phase step for DCT/DST-5
}
// temporal algorithm
static inline int estimate_radius(double s)
{
	//return (s<4.0) ? int(3.3333*s-0.3333+0.5) : int(3.4113*s-0.6452+0.5); //K==3
	return (s<4.0) ? int(3.0000*s-0.2000+0.5) : int(3.0000*s+0.5); //K==2
}
static inline std::vector<double> gen_spectrum(int K,double s,int r)
{
	const double omega=phase(r);
	std::vector<double> spect(K);
	for(int k=1;k<=K;k++)
		spect[k-1]=2.0*exp(-0.5*omega*omega*s*s*k*k);
	return spect;
}
static inline std::vector<double> build_lookup_table(int r,std::vector<double>& spect)
{
	const int K=spect.size();
	const double omega=phase(r);
	std::vector<double> table(K*(1+r));
	for(int u=0;u<=r;++u)
		for(int k=1;k<=K;++k)
			table[K*u+k-1]=cos(omega*k*u)*spect[k-1];
	return table;
}

//==================================================================================================

template <typename T,int CH,int K>
static inline void apply_spatial_gauss_x(int w,int h,T* src,T* dst,double s)
{
	throw std::invalid_argument("Unsupported parameters!");
}
template <typename T,int CH,int K>
static inline void apply_spatial_gauss_y(int w,int h,T* src,T* dst,double s)
{
	throw std::invalid_argument("Unsupported parameters!");
}

//--------------------------------------------------------------------------------------------------

template<>
static inline void apply_spatial_gauss_x<double,4,2>(int w,int h,double* src,double* dst,double s)
{
	const int CH=4,K=2;
	
	const int r=estimate_radius(s);
	const double norm=1.0/(r+1+r);
	std::vector<double> spect=gen_spectrum(K,s,r);
	std::vector<double> table=build_lookup_table(r,spect);

	const double cf11=table[K*1+0]*2.0/spect[0], cfR1=table[K*r+0];
	const double cf12=table[K*1+1]*2.0/spect[1], cfR2=table[K*r+1];

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
			const double* pE=&p[CH*atE(x+r+1)];
			const double* pW=&p[CH*atW(x-r  )];
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
		q[CH*0+0]=norm*(dc0+a0+aa0); dA0=diff[CH*0+0]; dc0+=dA0;
		q[CH*0+1]=norm*(dc1+a1+aa1); dA1=diff[CH*0+1]; dc1+=dA1;
		q[CH*0+2]=norm*(dc2+a2+aa2); dA2=diff[CH*0+2]; dc2+=dA2;
		q[CH*0+3]=norm*(dc3+a3+aa3); dA3=diff[CH*0+3]; dc3+=dA3;
		
		// the other pixels (x=1,2,...,w-1)
		int x=1;
		while(true) // with 4-length ring buffer
		{
			q[CH*x+0]=norm*(dc0+b0+bb0); dB0=diff[CH*x+0]; delta0=dA0-dB0; dc0+=dB0; a0+=-cf11*b0+cfR1*delta0; aa0+=-cf12*bb0+cfR2*delta0;
			q[CH*x+1]=norm*(dc1+b1+bb1); dB1=diff[CH*x+1]; delta1=dA1-dB1; dc1+=dB1; a1+=-cf11*b1+cfR1*delta1; aa1+=-cf12*bb1+cfR2*delta1;
			q[CH*x+2]=norm*(dc2+b2+bb2); dB2=diff[CH*x+2]; delta2=dA2-dB2; dc2+=dB2; a2+=-cf11*b2+cfR1*delta2; aa2+=-cf12*bb2+cfR2*delta2;
			q[CH*x+3]=norm*(dc3+b3+bb3); dB3=diff[CH*x+3]; delta3=dA3-dB3; dc3+=dB3; a3+=-cf11*b3+cfR1*delta3; aa3+=-cf12*bb3+cfR2*delta3;
			x++; if(w<=x) break;

			q[CH*x+0]=norm*(dc0-a0-aa0); dA0=diff[CH*x+0]; delta0=dB0-dA0; dc0+=dA0; b0+=+cf11*a0+cfR1*delta0; bb0+=+cf12*aa0+cfR2*delta0;
			q[CH*x+1]=norm*(dc1-a1-aa1); dA1=diff[CH*x+1]; delta1=dB1-dA1; dc1+=dA1; b1+=+cf11*a1+cfR1*delta1; bb1+=+cf12*aa1+cfR2*delta1;
			q[CH*x+2]=norm*(dc2-a2-aa2); dA2=diff[CH*x+2]; delta2=dB2-dA2; dc2+=dA2; b2+=+cf11*a2+cfR1*delta2; bb2+=+cf12*aa2+cfR2*delta2;
			q[CH*x+3]=norm*(dc3-a3-aa3); dA3=diff[CH*x+3]; delta3=dB3-dA3; dc3+=dA3; b3+=+cf11*a3+cfR1*delta3; bb3+=+cf12*aa3+cfR2*delta3;
			x++; if(w<=x) break;

			q[CH*x+0]=norm*(dc0-b0-bb0); dB0=diff[CH*x+0]; delta0=dA0-dB0; dc0+=dB0; a0+=-cf11*b0-cfR1*delta0; aa0+=-cf12*bb0-cfR2*delta0;
			q[CH*x+1]=norm*(dc1-b1-bb1); dB1=diff[CH*x+1]; delta1=dA1-dB1; dc1+=dB1; a1+=-cf11*b1-cfR1*delta1; aa1+=-cf12*bb1-cfR2*delta1;
			q[CH*x+2]=norm*(dc2-b2-bb2); dB2=diff[CH*x+2]; delta2=dA2-dB2; dc2+=dB2; a2+=-cf11*b2-cfR1*delta2; aa2+=-cf12*bb2-cfR2*delta2;
			q[CH*x+3]=norm*(dc3-b3-bb3); dB3=diff[CH*x+3]; delta3=dA3-dB3; dc3+=dB3; a3+=-cf11*b3-cfR1*delta3; aa3+=-cf12*bb3-cfR2*delta3;
			x++; if(w<=x) break;

			q[CH*x+0]=norm*(dc0+a0+aa0); dA0=diff[CH*x+0]; delta0=dB0-dA0; dc0+=dA0; b0+=+cf11*a0-cfR1*delta0; bb0+=+cf12*aa0-cfR2*delta0;
			q[CH*x+1]=norm*(dc1+a1+aa1); dA1=diff[CH*x+1]; delta1=dB1-dA1; dc1+=dA1; b1+=+cf11*a1-cfR1*delta1; bb1+=+cf12*aa1-cfR2*delta1;
			q[CH*x+2]=norm*(dc2+a2+aa2); dA2=diff[CH*x+2]; delta2=dB2-dA2; dc2+=dA2; b2+=+cf11*a2-cfR1*delta2; bb2+=+cf12*aa2-cfR2*delta2;
			q[CH*x+3]=norm*(dc3+a3+aa3); dA3=diff[CH*x+3]; delta3=dB3-dA3; dc3+=dA3; b3+=+cf11*a3-cfR1*delta3; bb3+=+cf12*aa3-cfR2*delta3;
			x++; if(w<=x) break;
		}
	}
}

//--------------------------------------------------------------------------------------------------

template<>
static inline void apply_spatial_gauss_y<double,4,2>(int w,int h,double* src,double* dst,double s)
{
	const int CH=4,K=2;

	const int r=estimate_radius(s);
	const double norm=1.0/(r+1+r);
	std::vector<double> spect=gen_spectrum(K,s,r);
	std::vector<double> table=build_lookup_table(r,spect);

	const double cf11=table[K*1+0]*2.0/spect[0], cfR1=table[K*r+0];
	const double cf12=table[K*1+1]*2.0/spect[1], cfR2=table[K*r+1];
	
	std::vector<double> workspace(CH*w*(2*K+2)); // work space to keep raster scanning

	// preparing initial entries
	for(int cx=0;cx<CH*w;++cx)
	{
		double* ws=&workspace[cx*(2*K+2)];
		ws[0]=src[cx];
		ws[1]=src[cx+CH*w]*table[0]; ws[2]=src[cx]*table[0];
		ws[3]=src[cx+CH*w]*table[1]; ws[4]=src[cx]*table[1];
	}
	for(int v=1;v<=r;++v)
	{
		for(int cx=0;cx<CH*w;++cx)
		{
			const double sum0=src[cx+CH*w*atN(0-v)]+src[cx+CH*w*(0+v)];
			const double sum1=src[cx+CH*w*atN(1-v)]+src[cx+CH*w*(1+v)];
			double* ws=&workspace[cx*(2*K+2)];
			ws[0]+=sum0;
			ws[1]+=sum1*table[K*v+0]; ws[2]+=sum0*table[K*v+0];
			ws[3]+=sum1*table[K*v+1]; ws[4]+=sum0*table[K*v+1];
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
			q[cx]=norm*(ws[0]+ws[2]+ws[4]);
			diff=p1S[cx]-p0N[cx];
			ws[0]+=diff;
			ws[5]=diff;
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
			q[cx]=norm*(ws[0]+ws[1]+ws[3]);

			diff=p1S[cx]-p0N[cx];
			delta=diff-ws[5];
			ws[0]+=diff;
			ws[2]=cfR1*delta+cf11*ws[1]-ws[2];
			ws[4]=cfR2*delta+cf12*ws[3]-ws[4];
			ws[5]=diff;
		}
		y++; if(h<=y) break;
		
		q=&dst[CH*w*y];
		p0N=&src[CH*w*atN(y-r  )];
		p1S=&src[CH*w*atS(y+r+1)];
		for(int cx=0;cx<CH*w;++cx)
		{
			double* ws=&workspace[cx*(2*K+2)];
			q[cx]=norm*(ws[0]+ws[2]+ws[4]);

			diff=p1S[cx]-p0N[cx];
			delta=diff-ws[5];
			ws[0]+=diff;
			ws[1]=cfR1*delta+cf11*ws[2]-ws[1];
			ws[3]=cfR2*delta+cf12*ws[4]-ws[3];
			ws[5]=diff;
		}
		y++; if(h<=y) break;
	}
}

//==================================================================================================

template <typename T,int CH,int K>
void apply_spatial_gauss(int w,int h,T* src,T* dst,double sx,double sy)
{
	if(w<=4.0*sx || h<=4.0*sy)
		throw std::invalid_argument("\'sx\' and \'sy\' should be less than about w/4 or h/4!");
		
	// filtering is skipped if s==0.0
	if(sx==0.0 && sy==0.0)
		return;
	else if(sx==0.0)
		apply_spatial_gauss_y<T,CH,K>(w,h,src,dst,sy);
	else if(sy==0.0)
		apply_spatial_gauss_x<T,CH,K>(w,h,src,dst,sx);
	else
	{
		apply_spatial_gauss_y<T,CH,K>(w,h,src,dst,sy);
		apply_spatial_gauss_x<T,CH,K>(w,h,dst,dst,sx); // only filter_gauss_x() allows in-place filtering.
	}
}

// OpenCV2 interface for easy function call
void apply_spatial_gauss(const cv::Mat& src,cv::Mat& dst,double sx,double sy)
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
//	case CV_32FC1: apply_spatial_gauss< float,1,2>(src.cols,src.rows,reinterpret_cast< float*>(src.data),reinterpret_cast< float*>(dst.data),sx,sy); break;
//	case CV_32FC4: apply_spatial_gauss< float,4,2>(src.cols,src.rows,reinterpret_cast< float*>(src.data),reinterpret_cast< float*>(dst.data),sx,sy); break;
//	case CV_64FC1: apply_spatial_gauss<double,1,2>(src.cols,src.rows,reinterpret_cast<double*>(src.data),reinterpret_cast<double*>(dst.data),sx,sy); break;
	case CV_64FC4: apply_spatial_gauss<double,4,2>(src.cols,src.rows,reinterpret_cast<double*>(src.data),reinterpret_cast<double*>(dst.data),sx,sy); break;
	default:
		throw std::invalid_argument("Unsupported element type or channel!");
		break;
	}
}

//==================================================================================================
