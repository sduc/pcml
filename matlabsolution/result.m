

clear all;
load('Weights2\earlyStopNeuron30.mat','w');
fid = fopen('data2\data24x24SubsetValTestPp.mat','r');
ga = ones(19440,24*24);
for i = 1:19440
ga(i,:) = fread(fid, 576,'double')';
end
% for i = 1:9720
% fread(fid, 576,'double')';
% end
% ga = ones(9720,24*24);
% for i = 1:9720
% ga(i,:) = fread(fid, 576,'double')';
% end

fclose(fid);
% ga = preprocessMeanVar(ga);

t= repmat([1 1 4 4],1,4860);

a = 0;
t1=[0 0];
res1 = 0;
yfinal = zeros(1,2);
MSEs = ones(1,9720);
% for i = 1:19440
for i = 1:9720

        a =[ga(i,:) 1];
        a1 = a*w{1};
        a2 = a*w{2};
        y = [a1./(1+exp(-a2)), 1];
        y1 = y*w{3};
        y2 = round(1./(1+exp(-y1)));
        
        if nnz(round(y2)) ~=1
            [val index] = max(y2);
            yfinal(index)=1;
        else
            yfinal = round(y2);
        end
    if t(i)==1
        t1=[1 0];
    elseif t(i)==4
        t1=[0 1];
    end
    
    if any(t1~=y2)
        res1 = res1+1;
    end
               MSEs(i) = 0.5*sum((t1 - y2).^2);
end
res1
r=(res1*100)/19440

mse = (1/19440)*sum(MSEs)










