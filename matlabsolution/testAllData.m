
clear all;

fid = fopen('data\data24x24Preprocessed.mat','r');

X = ones(48600,576);
for i = 1:48600 
X(i,:) = fread(fid,24*24,'double')';
end
fclose(fid);


fid2 = fopen('data\data24x24ValAndTestADPp.mat','r');
val = ones(24300,576);
for i = 1:24300
    val(i,:) = fread(fid2,24*24,'double')';
end
fclose(fid2);

X = preprocessMeanVar(X);
T = repmat([1 0 0 0 0;...
    1 0 0 0 0;...
    0 1 0 0 0;...
    0 1 0 0 0;...
    0 0 1 0 0;...
    0 0 1 0 0;...
    0 0 0 1 0;...
    0 0 0 1 0;...
    0 0 0 0 1;...
    0 0 0 0 1],4860,1);


weights = backPropMultiOutputAllData([576,10,5],0.0001,0.00001,0.9,X,T,val);






