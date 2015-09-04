function dst = o1_spat_filt(src,sigmaS)
% o1_spat_filt	O(1) spatial filter
%	src    : Source image
%	sigmaS : Scale of Gaussian spatial kernel
%
% This code implements the algorithm of the following paper. Please cite it in 
% your paper if your research uses this code.
%  + K. Sugimoto and S. Kamata: "Efficient constant-time Gaussian filtering with
%    sliding DCT/DST-5 and dual-domain error minimization", ITE Trans. Media 
%    Technol. Appl., vol. 3, no. 1, pp. 12-21 (Jan. 2015).
