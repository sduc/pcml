
clear all;

% Test for single Xor output

% X = [0,1;1,0;1,1;0,0];
% T = [ 1 1 0 0]';
% weights = backProp4XOR(0.0001,0.01,0.9, X, T);
% r = [1 1 1]*weights(:,1:4);
% 
% re = r(1:2)./(1+exp(-r(3:4)));
% res = [re 1]* weights(:,5);
% result = 1/(1+exp(-res))





% Test for Xor double output

% X = [0,1;1,0;1,1;0,0];
% T = [ 1 0; 1 0; 0 1; 0 1];
% weights = backPropXorMO([2,2,2],0.0001,0.001,0.99, X, T);
% 
% r1 = [1 1 1]*weights{1};
% r2 = [1 1 1]*weights{2};
% 
% re = r1./(1+exp(-r2));
% res = [re 1]* weights{3};
% result = 1./(1+exp(-res))




% % Test for Xor double output
% 
% X = [0,1;1,0;1,1;0,0];
% T = [ 1 0; 1 0; 0 1; 0 1];
% eta = [0.1,0.01,0.001];
% mu = [0.1,0.5,0.9, 0.99];
% hold on;
% for i = eta
%     for j = mu
% weights = backPropXorMO([2,2,2],0.0001,i,j, X, T);
% drawnow
%     end
% end
% r1 = [1 1 1]*weights{1};
% r2 = [1 1 1]*weights{2};
% 
% re = r1./(1+exp(-r2));
% res = [re 1]* weights{3};
% result = 1./(1+exp(-res))

% Test for Xor double output

X = [0,1;1,0;1,1;0,0];
T = [ 1 0; 1 0; 0 1; 0 1];
a = 0;
t1=[0 0];
res1 = 0;

% hold on;
% for j = [0.1,0.01, 0.001,0.0001 ]

    weights = backPropXorMO([2,2,2],0.0001,0.1,0.99, X, T);
%     drawnow
    for i = 1:4

        a =[X(i,:) 1];
        a1 = a*weights{1};
        a2 = a*weights{2};
        y = [a1./(1+exp(-a2)), 1];
        y1 = y*weights{3};
        y2 = round(1./(1+exp(-y1)))


        if T(i)~=y2
            res1 = res1+1;
        end

    end
% end

res1



