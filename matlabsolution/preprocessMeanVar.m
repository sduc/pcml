


function out = preprocessMeanVar(data)
% Data is a matrix of images in vector form. Each image is a row of the
% matrix.
% Out is of the same dimension than Data, but is now 0-mean and unit
% standard deviation.

M = size(data,1);
mu = mean(data);
sigma = std(data);

sigma(sigma < 0.00001) = 1;% make sure there are no divisions by zero
mu(sigma < 0.00001) = 0;

meanMat = repmat(mu,M,1);
stdMat = repmat(sigma,M,1);

out = (data - meanMat)./stdMat;

end



