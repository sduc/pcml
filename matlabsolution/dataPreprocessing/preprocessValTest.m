
% Preprocess for Validation set and test set.


fid = fopen('..\data2\data24x24Subset.mat','r');

X = ones(19440,576);
for i = 1:19440 
X(i,:) = fread(fid,24*24,'double')';
end
fclose(fid);

mu = mean(X);
sigma = std(X);

sigma(sigma < 0.00001) = 1;% make sure there are no divisions by zero
mu(sigma < 0.00001) = 0;

fid2 = fopen('..\data2\data24x24SubsetValTest.mat','r');
fid3 = fopen('..\data2\data24x24SubsetValTestPp.mat', 'w+');

im = ones(1,576);
for j = 1:19440
    im = fread(fid2,24*24,'double')';
    fwrite(fid3,((im - mu)./sigma)','double');
end

fclose(fid2);
fclose(fid3);