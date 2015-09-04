function dst = bi_filt_comp(src,sigmaS,sigmaR,tol)
% bi_filt_comp	Compressive bilateral filter
%	src    : Source image
%	sigmaS : Scale of Gaussian spatial kernel
%	sigmaR : Scale of Gaussian range kernel
%	tol    : Tolerance
%
% This code implements the algorithm of the following paper. Please cite it in 
% your paper if your research uses this code.
%  + K. Sugimoto and S. Kamata: "Compressive bilateral filtering", IEEE Trans.
%    Image Process., vol. 24, no. 11, pp. 3357-3369 (Nov. 2015).
%
% See also o1filter

	%% Estimating an optimal K
	xi = erfcinv(tol^2);
	K = ceil(xi*xi/(2.0*pi)+xi/(2.0*pi*sigmaR)-0.5);

	%% Estimating an optimal T
	kappa = pi*(2*K+1);
	df = @(t) kappa*exp(-((t-1.0)/sigmaR)^2)-exp(-(kappa*sigmaR/t)^2)*(kappa*sigmaR/t)^2;
	t1 = sigmaR*xi+1.0;
	t2 = kappa*sigmaR/xi;
	% It is better to slightly extend the original search domain D
	% because it might uncover the minimum of E(T) due to approximate error.
	MAGICNUM = 0.03;
	T = fzero(df,[t1 t2+MAGICNUM]);
	
	%% DC component
	[numer,ratio] = o1_spat_filt(src,sigmaS);
	denom = ones(size(src))*ratio;

	%% AC components
	omega = 2.0*pi/T;
 	for k = 1:K
		Iphase = omega*k*src;
		Ic = cos(Iphase);
		Is = sin(Iphase);
		CIc  = o1_spat_filt(Ic     ,sigmaS);
		CIs  = o1_spat_filt(Is     ,sigmaS);
		CIcp = o1_spat_filt(Ic.*src,sigmaS);
		CIsp = o1_spat_filt(Is.*src,sigmaS);

 		ak = 2.0*exp(-0.5*(omega*k*sigmaR)^2);
 		numer = numer +ak*(Ic.*CIcp+Is.*CIsp);
 		denom = denom +ak*(Ic.*CIc +Is.*CIs );
	end
	
	%% Generating smoothed image
	dst = numer./denom;
end
