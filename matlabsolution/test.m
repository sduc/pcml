
clear all;

% weights = backProptest([9216,10,1],0.0001,0.01,0.9)



% % fid = fopen('subset.mat','r');
% % 
% % X = ones(1000,9216);
% % for i = 1:1000 
% % X(i,:) = fread(fid,96*96)';
% % end
% % fclose(fid);
% % 
% % X = preprocessMeanVar(X);
% % T = repmat([1 0;0 1],500,1);
% % 
% % 
% % weights = backPropMultiOutput([9216,10,2],0.0001,0.01,0.9,X,T);

% fid = fopen('data\data24x24Subset.mat','r');
% 
% X = ones(19440,576);
% for i = 1:19440 
% X(i,:) = fread(fid,24*24)';
% end
% fclose(fid);
% 
% X = preprocessMeanVar(X);
% hist(X)
% imagesc(reshape(X(1,:),24,24)');
% imshow(reshape(X(1,:),24,24)',[0 255]);
% T = repmat([1 0; 1 0; 0 1;0 1],4860,1);
% 
% 
% weights = backPropMultiOutput([576,10,2],0.0001,0.0001,0.9,X,T);


fid = fopen('data2\data24x24Subset.mat','r');
X = ones(19440,576);
% X = ones(9720,576);
for i = 1:19440   
X(i,:) = fread(fid,24*24,'double')';
end
fclose(fid);


fid2 = fopen('data2\data24x24SubsetValTestPp.mat','r');
val = ones(9720,576);
for i = 1:9720
    val(i,:) = fread(fid2,24*24,'double')';
end
fclose(fid2);
% figure(2)
% imagesc(reshape(val(1,:),24,24)');

X = preprocessMeanVar(X);
T = repmat([1 0; 1 0; 0 1;0 1],4860,1);
% T = repmat([1 0; 0 1],4860,1);

weights = backPropMultiOutput([576,10,2],0.0001,0.00001,0.9,X,T,val);
save('Weights\MSEStopNeuron10.mat','weights');




