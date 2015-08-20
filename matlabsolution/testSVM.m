
clear all;
addpath('..');

fid = fopen('..\data2\data24x24Subset.mat','r');
ts = repmat([-1;-1;1;1],500,1);

elem = randperm(19440);


Xs = ones(19440,576);
for i = 1:19440  
Xs(i,:) = fread(fid,24*24,'double')';
end
fclose(fid);

X = Xs(elem(1:2000),:);

fid2 = fopen('..\data2\data24x24SubsetValTestPp.mat','r');
V = ones(2000,576);
for i = 1:2000 
V(i,:) = fread(fid2,24*24,'double');
end
fclose(fid2);

X = preprocessMeanVar(X);
K = ones(2000,2000);



for h = 1:2:10
n = 100;
C = h
eta = 0.01;
err= zeros(1,n);
err2= zeros(1,n);


for j = 1:n
    sigma2 = j;

    for i = 1:100
        dist = X - repmat(V(i,:),100,1);
        dist = sum(dist.^2,2); % corresponds to ||x - xnewimage||^2
        K(:,i) = exp(-dist/(2*sigma2));
    end

    Kernel = kern(X,sigma2);

    out = svm(X,T,eta,C,Kernel);
    alpha = out{1};
    v = out{2};


    for i = 1:100
        xOut = sign((alpha.*T)'* Kernel(:,i)-v);
        err(j) = err(j)+ abs(xOut-T(i))/2;
    end
    for i = 1:100
        xOut = sign((alpha.*T)'* K(:,i)-v);
        err2(j) = err2(j)+ abs(xOut-T(i))/2;
    end
end

figure(h);
plot(1:n, err,'-r',1:n,err2,'-b');

end


