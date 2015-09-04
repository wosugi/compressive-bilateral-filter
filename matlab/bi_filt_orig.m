function dst = bi_filt_orig(src,sigmaS,sigmaR)
% bi_filt_orig	Original bilateral filter
%	src    : Source image
%	sigmaS : Scale of Gaussian spatial kernel
%	sigmaR : Scale of Gaussian range kernel

	%% Generating Gaussian spatial kernel
	winsz = ceil(4.0*sigmaS);
	[u,v] = meshgrid(-winsz:+winsz,-winsz:+winsz);
	kernelS = exp(-0.5*(u.^2+v.^2)/(sigmaS^2));

	%% Filtering
	[h,w,ch] = size(src);
	dst = zeros(h,w,ch);
	wb = waitbar(0,'Original bilateral filtering ...');
	for y = 1:h
		for x = 1:w
 			u1 = max(-winsz,1-x); u2 = min(+winsz,w-x);
 			v1 = max(-winsz,1-y); v2 = min(+winsz,h-y);
 			
 			for c = 1:ch
 				numer = 0.0;
 				denom = 0.0;
 				for v = v1:v2
 					for u = u1:u2
 						dr = src(y+v,x+u,c)-src(y,x,c);
 						kernelR = exp(-0.5*(dr/sigmaR)^2);
 						g = kernelS(v+winsz+1,u+winsz+1)*kernelR;
 						numer = numer+g*src(y+v,x+u,c);
 						denom = denom+g;
 					end
 				end
 				dst(y,x,c) = numer/denom;
 			end
		end
		waitbar(y/h,wb);
	end
	close(wb);
end
