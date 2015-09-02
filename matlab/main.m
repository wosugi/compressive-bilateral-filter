
%% Parameters
sigmaS = 2.0; % Scale of Gaussian spatial kernel
sigmaR = 0.1; % Scale of Gaussian range kernel
tol = 0.1; % Tolerance

% Source image has to have dynamic range [0,1].
pathI = 'lenna.png';
src = im2double(imread(pathI));

%% Original bilateral filter
fprintf('[Original bilateral filter]\n');
tic;
dst0 = bf(src,sigmaS,sigmaR);
toc

%% Compressive bilateral filter
fprintf('[Compressive bilateral filter]\n');
tic;
dst1 = cbf(src,sigmaS,sigmaR,tol);
toc

%% Approximate accuracy
snr = @(a,b,maxval) 10*log10(maxval^2/mean((a(:)-b(:)).^2));
fprintf('SNR:  %f\n',snr(dst0,dst1,1.0));

%imwrite(dst0,'bf.png');
%imwrite(dst1,'cbf.png');
figure(1), imshow([dst0,dst1]);
