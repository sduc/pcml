% Preprocess for whole data.


fid = fopen('..\data2\data24x24.mat','r');

X = ones(1,576);
m = zeros(1,576);
m2 = zeros(1,576);
for i = 1:48600 
X = fread(fid,24*24,'double')';
m = m + X;
m2 = m2 + X.^2;
end
fclose(fid);
mu = m./48600;
mom2 = m2./48600;
sigma = sqrt((mu.^2) - mom2);

sigma(sigma < 0.0001) = 1;% make sure there are no divisions by zero
mu(sigma < 0.0001) = 0;

fid = fopen('..\data2\data24x24.mat','r');
fid2 = fopen('..\data2\data24x24Preprocessed.mat','w+');

im = ones(1,576);
for i = 1:48600
    im = fread(fid,24*24,'double')';
    fwrite(fid2,(im - mu)./sigma,'double');
end
fclose(fid);
fclose(fid2);




fid = fopen('..\data2\data24x24ValTestAD.mat','r');
fid2 = fopen('..\data2\data24x24ValTestADPp.mat', 'w+');

im = ones(1,576);
for j = 1:48600
    im = fread(fid,24*24,'double')';
    fwrite(fid2,(im - mu)./sigma,'double');
end

fclose(fid);
fclose(fid2);