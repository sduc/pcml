
% SVM on AND 
% *****************************
% X = [1 0;0 1;1 1; 0 0];%AND
% T = [-1;-1;1;-1];
% 
% % X=[0.9 0.1; 0.9 0.3; 0.77 0.75;0.1 0.2]; 
% V=[0.9 0.1; 1 0.4; 0.75 0.75;0.1 0.2]; 
% 
% C = 10;
% eta = 0.5;
% 
% K=X*V';
% Kernel = X*X';
% 
% out = svm(X,T,eta,C,Kernel);
% alpha = out{1};
% v = out{2};
% 
% err = 0;
% err2 = 0;
% 
% for i = 1:4
%     xOut = sign((alpha.*T)'* Kernel(:,i)-v);
%     err = err+ abs(xOut-T(i))/2;
% end
% 
% for i = 1:4
%     xOut = sign((alpha.*T)'* K(:,i)-v);
%     err2 = err2+ abs(xOut-T(i))/2;
% end
% fprintf('Nb training errors:');
% disp(err)
% fprintf('Nb validation errors:');
% disp(err)


% SVM on XOR
%****************************************

% X = [1 0;0 1;1 1; 0 0];%XOR
% T = [1;1;-1;-1];
% % X=[0.9 0.1; 0.9 0.3; 0.77 0.75;0.1 0.2]; 
% V=[0.9 0.1; 1 0.4; 0.75 0.75;0.1 0.2]; 
% 
% 
% K= ones(4,4);
% 
% err= zeros(1,100);
% err2 =zeros(1,100);
% for j = 1:100
%     C = 30.5;
%     eta = 0.01;
%     sigma2 = j;
% 
%     for i = 1:4
%         dist = X - repmat(V(i,:),4,1);
%         dist = sum(dist.^2,2); % corresponds to ||x - xnewimage||^2
%         K(:,i) = exp(-dist/(2*sigma2));
%     end
% 
%     Kernel = kern(X,sigma2);
% 
%     out = svm(X,T,eta,C,Kernel);
%     alpha = out{1};
%     v = out{2};
% 
%     for i = 1:4
%         xOut = sign((alpha.*T)'* Kernel(:,i)-v);
%         err(j) = err(j)+ abs(xOut-T(i))/2;
%     end
%     
%     for i = 1:4
%         xOut = sign((alpha.*T)'* K(:,i)-v);
%         err2(j) = err2(j)+ abs(xOut-T(i))/2;
%     end
%     
% end
% plot(1:100, err,'-r',1:100,err2,'-b');


% SVM on NORB
% ******************************************************
clear all;
addpath('..');

fid = fopen('..\data2\data24x24Subset.mat','r');

X = ones(2000,576);
for i = 1:2000 
X(i,:) = fread(fid,24*24,'double')';
end
fclose(fid);
X = preprocessMeanVar(X);

fid2 = fopen('..\data2\data24x24SubsetValTestPp.mat','r');

V = ones(2000,576);
for i = 1:2000 
V(i,:) = fread(fid2,24*24,'double');
end
fclose(fid2);

T = repmat([-1;-1;1;1],500,1);
K = ones(2000,2000);

n = 5; %nb of different sigma squares to test
val = 13:18;

C = 0.5;
eta = 0.1;
err= zeros(1,n);
err2= zeros(1,n);


% main test
for j = 1:n
    sigma2 = val(j)

    for i = 1:2000
        dist = X - repmat(V(i,:),2000,1);
        dist = sum(dist.^2,2); % corresponds to ||x - xnewimage||^2
        K(:,i) = exp(-dist/(2*sigma2));
    end

    Kernel = kern(X,sigma2);

    out = svm(X,T,eta,C,Kernel);
    alpha = out{1};
    v = out{2};


    for i = 1:2000
        xOut = sign((alpha.*T)'* Kernel(:,i)-v);
        err(j) = err(j)+ abs(xOut-T(i))/2;
    end
    for i = 1:2000
        xOut = sign((alpha.*T)'* K(:,i)-v);
        err2(j) = err2(j)+ abs(xOut-T(i))/2;
    end
end

plot(1:n, err,'-r',1:n,err2,'-b');


% Test the SVM on sets
% **************************************************************
% clear all;
% addpath('..');
% load('alphaOpt.mat','alpha');
% fid = fopen('..\data2\data24x24Subset.mat','r');
% 
% X = ones(2000,576);
% for i = 1:2000 
% X(i,:) = fread(fid,24*24,'double')';
% end
% fclose(fid);
% X = preprocessMeanVar(X);
% 
% % Uncomment to test validation set********************
% 
% % fid2 = fopen('..\data2\data24x24SubsetValTestPp.mat','r');
% % 
% % for i = 1:9720 
% % fread(fid2,24*24,'double');
% % end
% % 
% % V = ones(9720,576);
% % for i = 1:9720 
% % V(i,:) = fread(fid2,24*24,'double');
% % end
% % fclose(fid2);
% % ****************************************************
% 
% % Uncomment to test validation set********************
% fid2 = fopen('..\data2\data24x24Subset.mat','r');
% 
% V = ones(19440,576);
% for i = 1:19440 
% V(i,:) = fread(fid2,24*24,'double');
% end
% fclose(fid2);
% V = preprocessMeanVar(V);
% % ****************************************************
% 
% T = repmat([-1;-1;1;1],500,1);
% T2 = repmat([-1;-1;1;1],4860,1);
% 
% v = 0.8381;
% err = 0;
% for i = 1:19440
%     xOut = sign((alpha.*T)'* kern(X,V(i,:),34)-v);
%     err = err+ abs(xOut-T2(i))/2;
% end
% 
% fprintf('Nb errors');
% disp(err);



