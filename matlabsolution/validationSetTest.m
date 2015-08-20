

function mse = validationSetTest(w,T, valSet)

len = size(valSet,1);

MSEs = ones(1,len);

for i = 1:len

    a =[valSet(i,:) 1];
    a1 = a*w{1};
    a2 = a*w{2};
    a2(a2 > 10) = 10; % avoid the too large or too small exponential
    a2(a2 < -10) = -10;
    
    y1 = [a1./(1+exp(-a2)), 1];
    
    y2 = y1*w{3};
    y2(y2 > 10) = 10;
    y2(y2 < -10) = -10;
    y3 = 1./(1+exp(-y2));

    MSEs(i) = 0.5*sum((T(i,:) - y3).^2);

end

mse = (1/len)*sum(MSEs);

end
% 
% clear all;
% 
% load('Weights2\earlyStopNeuron40LR0.000001Mom99.mat','w');
% fid = fopen('data2\data24x24SubsetValTestPp.mat','r');
% % len = size(valSet,1);
% for i = 1:9720
% fread(fid, 576,'double');
% end
% ga = ones(9720,24*24);
% for i = 1:9720
% ga(i,:) = fread(fid, 576,'double')';
% end
% fclose(fid);
% 
% T = repmat([1 0;1 0;0 1;0 1],4860,1);
% 
% MSEs = ones(1,9720);
% 
% for i = 1:9720
% 
%     a =[ga(i,:) 1];
%     a1 = a*w{1};
%     a2 = a*w{2};
% %     a1 = a*minW{1};
% %     a2 = a*minW{2};
%     a2(a2 > 10) = 10; % avoid the too large or too small exponential
%     a2(a2 < -10) = -10;
%     
%     y1 = [a1./(1+exp(-a2)), 1];
%     
% %         y2 = y1*minW{3};
%     y2 = y1*w{3};
%     y2(y2 > 10) = 10;
%     y2(y2 < -10) = -10;
%     y3 = 1./(1+exp(-y2));
% 
%     MSEs(i) = 0.5*sum((T(i,:) - y3).^2);
% 
% end
% 
% mse = (1/9720)*sum(MSEs)
