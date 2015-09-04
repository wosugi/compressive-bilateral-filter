
%% Parameters
pathI = 'lenna.png';
sigmaS = 2.0; % Scale of Gaussian spatial kernel
sigmaR = 0.1; % Scale of Gaussian range kernel
tol = 0.1; % Tolerance
fprintf('[Parameters]\n');
fprintf('Path=%s\n',pathI);
fprintf('SigmaS=%f  SigmaR=%f  Tolerance=%f\n',sigmaS,sigmaR,tol);

%% Loading source image
src = im2double(imread(pathI)); % Dynamic range [0,1]

%% Building MEX
if exist('o1_spat_filt','file') ~= 3
	fprintf('[MEX building]\n');
	mex('o1_spat_filt.cpp');
end

%% Original bilateral filter
fprintf('[Original bilateral filter]\n');
tic;
dst0 = bi_filt_orig(src,sigmaS,sigmaR);
toc

%% Compressive bilateral filter
fprintf('[Compressive bilateral filter]\n');
tic;
dst1 = bi_filt_comp(src,sigmaS,sigmaR,tol);
toc

%% Approximate accuracy
psnr = @(a,b,maxval) 10*log10(maxval^2/mean((a(:)-b(:)).^2));
fprintf('PSNR:  %f\n',psnr(dst0,dst1,1.0));

%imwrite(dst0,'dst_bi_filt_orig.png');
%imwrite(dst1,'dst_bi_filt_comp.png');
figure(1), imshow([dst0,dst1]);
