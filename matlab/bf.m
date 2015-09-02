function dst = bf(src,sigmaS,sigmaR)
% bf	Original bilateral filter
%	src    : Source image (with dynamic range [0,1])
%	sigmaS : Scale of Gaussian spatial kernel
%	sigmaR : Scale of Gaussian range kernel

	%% Generating Gaussian spatial kernel
	R = ceil(4.0*sigmaS);
	[u,v] = meshgrid(-R:+R,-R:+R);
	kernelS = exp(-0.5*(u.^2+v.^2)/(sigmaS^2));

	%% Filtering
	[h,w,ch] = size(src);
	dst = zeros(h,w,ch);
	wb = waitbar(0,'Original bilateral filtering ...');
	for y = 1:h
		for x = 1:w
 			u1 = max(-R,1-x); u2 = min(+R,w-x);
 			v1 = max(-R,1-y); v2 = min(+R,h-y);
 			
 			for c = 1:ch
 				numer = 0.0;
 				denom = 0.0;
 				for v = v1:v2
 					for u = u1:u2
 						dr = src(y+v,x+u,c)-src(y,x,c);
 						kernelR = exp(-0.5*(dr/sigmaR)^2);
 						g = kernelS(v+R+1,u+R+1)*kernelR;
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
