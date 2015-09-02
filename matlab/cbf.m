function dst = cbf(src,sigmaS,sigmaR,tol)
% cbf	Compressive bilateral filter
%	src    : Source image (with dynamic range [0,1])
%	sigmaS : Scale of Gaussian spatial kernel
%	sigmaR : Scale of Gaussian range kernel
%	tol    : Tolerance
%
% This code implements the algorithm of the following paper. Please cite it in 
% your paper if your research uses this code.
%  + K. Sugimoto and S. Kamata: "Compressive bilateral filtering", IEEE Trans.
%    Image Process., vol. 24, no. 11, pp. 3357-3369 (Nov. 2015).
%
% See also ctfilter

	%% Estimating an optimal K
	xi = erfcinv(tol^2);
	K = ceil(xi*xi/(2.0*pi)+xi/(2.0*pi*sigmaR)-0.5);

	%% Estimating an optimal T
	syms x;
	kappa = pi*(2*K+1);
	phiphi = ((x-1.0)/sigmaR)^2;
	psipsi = (kappa*sigmaR/x)^2;
	df = (kappa*exp(-phiphi)-psipsi*exp(-psipsi)==0);
	x1 = sigmaR*xi+1.0;
	x2 = sigmaR*kappa/xi;
	% It is better to slightly extend the original search domain D
	% because it might uncover the minimum of E(T) due to approximate error.
	MAGICNUM = 0.03;
	T = double(vpasolve(df,x,[x1 x2+MAGICNUM]));
	
	%% DC component
	[numer,ratio] = o1filter(src,sigmaS);
	denom = ones(size(src))*ratio;
	
	%% AC components
	omega = 2.0*pi/T;
 	for k = 1:K
		Iphase = src*omega*k;
		Ic = cos(Iphase);
		Is = sin(Iphase);
		CIc  = o1filter(Ic     ,sigmaS);
		CIs  = o1filter(Is     ,sigmaS);
		CIcp = o1filter(Ic.*src,sigmaS);
		CIsp = o1filter(Is.*src,sigmaS);

		ak = 2.0*exp(-0.5*(omega*k*sigmaR)^2);
		numer = numer +ak*(Ic.*CIcp+Is.*CIsp);
		denom = denom +ak*(Ic.*CIc +Is.*CIs );
	end
	
	%% Generating smoothed image
	dst = numer./denom;
	
	dst = max(dst,ones(size(dst))*0.0);
	dst = min(dst,ones(size(dst))*1.0);
end
