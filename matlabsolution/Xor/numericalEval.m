
% Numerical evaluation of The XOR back-propagation function
% Coordinates is a 1x2 vector of the coordinates in the weight matrice of
% the specific weight we want to test. WeightMat is the weight matrice we
% want to choose the weight from(1 to 3). Example: if I want to see the weight
% of the first neuron in input layer to the second neuron in the hidden
% layer for the first set of weights, I choose weightMat = 1 
% and coordinates = [1,2]. Mod refers to the 2 different
% output moodels available.
% 
%Backprop model1 mod = 1
%in1 = 1 o---o 
%          X  >o out1
%in2 = 2 o---o
%
%Backprop model2 mod = 2
%in1 = 1 o---o---o out1
%          X   X
%in2 = 2 o---o---o out2
%




function numericalEval(coordinates,weightMat,mod)

global w;
global v;
global y;

NpL = [2 2 mod];
layers = 3;
a = -0.1;
b = 0.1;
e = 0.00001;

if mod == 1
    t = 0;
elseif mod ==2
    t = [0 0];
end


%****** INITIALISATION OF THE WEIGHTS ******
w = cell(layers,1);
w{1} = a + (b-a).*rand(NpL(1)+1 , NpL(2)) ;%random value in interval [a,b]
w{2} = a + (b-a).*rand(NpL(1)+1 , NpL(2)) ;% Because we have h1 and h2
w{end} = a + (b-a).*rand(NpL(2)+1,NpL(end));

%****** PREALLOCATION ******
y = cell(layers,1);  % one y matrix for each layer
y{1} = ones(1,NpL(1)+1); % y{1} is the input + '1' for the bias node activation
y{2} = ones(1,NpL(2)+1); % inner layer includes a bias node 
y{end} = ones(1,NpL(end)); 

v = cell(layers,1); 
v{1} = ones(1, NpL(2));
v{2} = ones(1, NpL(2));
v{end} = ones(1, NpL(3));

delta = cell(3,1);
delta{1} = ones(1,NpL(2));
delta{2} = ones(1,NpL(2));
delta{3} = ones(1,NpL(end));

Dw = cell(layers,1);
E = ones(1,2);
%****************************



y{1} = [1 1 1];% just test for [1 1] because all the weights are turned on

%***** Forward Computation *****
err = fastForward(t);
%************************************


%***** Back-Propagation *****
delta{3} = err.*y{end}.*(1-y{end}); % delta k (has this simple form because it is a sigmoid function)!!!
delta{1} = (1./(1+exp(-v{2}))) .* (delta{3} * w{3}(1:end-1,:)');
delta{2} = (v{1}.*exp(-v{2}))./((1+exp(-v{2})).^2) .* (delta{3} * w{3}(1:end-1,:)');

Dw{1} = (y{1}')*delta{1};
Dw{2} = (y{1}')*delta{2};
Dw{3} = (y{2}')*delta{3};
%************************************


%***** Computation of E(deltaW + e) and E(deltaW - e) *****
w{weightMat}(coordinates(1),coordinates(2)) = w{weightMat}(coordinates(1),coordinates(2)) + e;
err = fastForward(t);
E(1) = 0.5*sum(err.^2);

w{weightMat}(coordinates(1),coordinates(2)) = w{weightMat}(coordinates(1),coordinates(2)) - 2*e;
err = fastForward(t);
E(2) = 0.5*sum(err.^2);
%************************************
format long e;
f =Dw{weightMat}(coordinates(1),coordinates(2));
ff = (E(1)-E(2))/(2*e);
difference = f+ff;

fprintf('\n*********************************************\n');
fprintf('Computed Weight Update & Numerical Evaluation \n');
fprintf('*********************************************\n');
disp([-Dw{weightMat}(coordinates(1),coordinates(2)) , (E(1)-E(2))/(2*e), difference]);


end

function err = fastForward(t)

global w;
global v;
global y;

y{1} = [1 1 1];
v{1} =  y{1}*w{1}; % correspond � vj(n)
v{2} =  y{1}*w{2}; % correspond � vj(n)

v{2}(v{2} > 10) = 10; % avoid the too large or too small exponential
v{2}(v{2} < -10) = -10;

y{2} = [v{1}./(1+exp(-v{2})), 1]; % dim( 1 x P) P = nb hidden nodes

v{3} = y{2}*w{end}; % correspond � vk(n)
v{3}(v{3} > 10) = 10;
v{3}(v{3} < -10) = -10;

y{end} = 1./(1+exp(-v{3})); % correspond � yk(n)

err = t - y{end};
end
   