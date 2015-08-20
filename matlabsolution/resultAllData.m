clear all;
load('Weights2\weightsESAd50LR000001.mat','w');
fid = fopen('data2\data24x24ValTestADPp.mat','r');
% ga = ones(19440,24*24);
% for i = 1:19440
% ga(i,:) = fread(fid, 576,'double')';
% end
% count = 0;
% while ~feof(fid)
%     fread(fid, 576,'double')';
%     count = count+1
% end

for i = 1:24300
    fread(fid, 576,'double');
end
ga = ones(24300,24*24);
for i = 1:24300
    ga(i,:) = fread(fid, 576,'double')';
end

fclose(fid);
% ga = preprocessMeanVar(ga);

t = repmat([1 0 0 0 0;...
    1 0 0 0 0;...
    0 1 0 0 0;...
    0 1 0 0 0;...
    0 0 1 0 0;...
    0 0 1 0 0;...
    0 0 0 1 0;...
    0 0 0 1 0;...
    0 0 0 0 1;...
    0 0 0 0 1],4860,1);

a = 0;
res1 = 0;
yfinal = zeros(1,5);
MSEs = ones(1,24300);

for i = 1:24300

    a =[ga(i,:) 1];
    a1 = a*w{1};
    a2 = a*w{2};
    y = [a1./(1+exp(-a2)), 1];
    y1 = y*w{3};
    y2 = 1./(1+exp(-y1));

    if nnz(round(y2)) ~=1
        [val index] = max(y2);
        yfinal(index)=1;
    else
        yfinal = round(y2);
    end
    if any(t(i,:)~=yfinal)
        res1 = res1+1;
    end
    MSEs(i) = 0.5*sum((t(i,:) - y2).^2);

end
res1
r=(res1*100)/24300
mse = (1/24300)*sum(MSEs)
