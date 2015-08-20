

function weights = backPropMultiOutput(NpL, sufMSE,LRate, Momentum,X,T,valSet)

[N P] = size(X); %P= dimension of the input vector, N= nb of points

layers = length(NpL); %NpL = neurons per layers (1 X nbLayers)
epoch = 0;
a = -0.1;
b = 0.1;
mse = 100;
VT = T;

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

prev_w = cell(layers,1);
prev_w{1} = zeros(NpL(1)+1, NpL(2));
prev_w{2} = zeros(NpL(1)+1, NpL(2));
prev_w{3} = zeros(NpL(2)+1,NpL(end));

MSEs = ones(1,N);
AvMSEs = zeros(1,1000);
validationMSEs = zeros(1,1000);
minMSE = 100;

while mse > sufMSE && epoch < 1000

    for k = 1:N

        y{1} = [X(k,:) 1];

        %************************************
        % Forward Computation
        %************************************
        v{1} =  y{1}*w{1}; % correspond à vj(n)
        v{2} =  y{1}*w{2}; % correspond à vj(n)

        v{2}(v{2} > 10) = 10; % avoid the too large or too small exponential
        v{2}(v{2} < -10) = -10;

        y{2} = [v{1}./(1+exp(-v{2})), 1]; % dim( 1 x P) P = nb hidden nodes

        v{3} = y{2}*w{end}; % corresponds to vk(n)
        v{3}(v{3} > 10) = 10;
        v{3}(v{3} < -10) = -10;

        y{end} = 1./(1+exp(-v{3})); % corresponds to yk(n)
        err = T(k,:) - y{end};   % is a vector 1x(size of output)

        %************************************
        % Back-Propagation
        %************************************
        delta{3} = err.*y{end}.*(1-y{end}); % delta k (has this simple form because it is a sigmoid function)!!!
        delta{1} = (1./(1+exp(-v{2}))) .* (delta{3} * w{3}(1:end-1,:)');
        delta{2} = (v{1}.*exp(-v{2}))./((1+exp(-v{2})).^2) .* (delta{3} * w{3}(1:end-1,:)');

        %****** Weight Update *******
        prev_w{1} = Momentum*prev_w{1} + LRate*(y{1}')*delta{1};
        w{1} = w{1} + prev_w{1};

        prev_w{2} = Momentum*prev_w{2} + LRate*(y{1}')*delta{2};
        w{2} = w{2} + prev_w{2};

        prev_w{3} = Momentum*prev_w{3} + LRate*(y{2}')*delta{3};
        w{3} = w{3} + prev_w{3};

        MSEs(k) = 0.5*sum(err.^2);
    end

    %Mean square error
    mse = (1/N)*sum(MSEs);
    AvMSEs(epoch+1) = mse;

    validationMSEs(epoch+1) = validationSetTest(w,VT,valSet); %Gets the validation error

    %gets the minimal validation error weights
    if validationMSEs(epoch+1)< minMSE
        minMSE = validationMSEs(epoch+1);
        minW = w;
    end

    % permutation of the points
    permutation = randperm(N);
    X = X(permutation ,:);
    T = T(permutation,:);

    epoch = epoch + 1
end
save('Weights\earlyStopNeuron10.mat','minW');
plot(1:1000,AvMSEs,'-*r',1:1000,validationMSEs,'-*b');
weights = w;
end








