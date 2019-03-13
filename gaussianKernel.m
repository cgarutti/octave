%GAUSSIANKERNEL returns a Gaussian kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim
function sim = gaussianKernel(x1, x2, sigma);
  x1 = x1(:); x2 = x2(:); %ensure that x1 and x2 are column vectors
  sim = exp(-((norm(x1-x2))^2)/(2*sigma^2)); %return Gaussian kernel
end;
