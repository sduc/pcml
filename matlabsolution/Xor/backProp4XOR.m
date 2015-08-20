

function weights = backProp4XOR(sufMSE,LRate, Momentum, X, T)

[N P] = size(X); %P= dimension of the input vector, N= nb of points

layers = 3;
epoch = 0;
a = -0.1;
b = 0.1;
mse = 100;
ga = zeros(1,30000);

%****** INITIALISATION OF THE WEIGHTS ******
w = cell(layers,1);
w{1} = a + (b-a).*rand(P+1,P) ;%random value in interval [a,b]
w{2} = a + (b-a).*rand(P+1,P) ;
w{end} = a + (b-a).*rand(P+1,1);

%****** PREALLOCATION ******
y = cell(layers - 1,1);  % one activation matrix for each layer
y{1} = [X ,ones(N,1)]; % a{1} is the input + '1' for the bias node activation
y{2} = ones(1,P+1); % inner layers include a bias node (P-by-Nodes+1) 

delta = cell(3,1);
delta{1} = ones(1,P);
delta{2} = ones(1,P);
delta{3} = ones(1,1);

prev_w = cell(layers,1);
prev_w{1} = zeros(P+1, P);
prev_w{2} = zeros(P+1, P);
prev_w{3} = zeros(P+1,1);

MSEs = ones(1,N);

while mse > sufMSE && epoch < 30000
    
    for k = 1:N
    
        
    %************************************
    % Forward Computation
    %************************************
    v1 =  y{1}(k,:)*w{1}; % correspond à vj(n)
    v2 =  y{1}(k,:)*w{2}; % correspond à vj(n)
    
    v2(v2 > 10) = 10; % avoid the too large or too small exponential
    v2(v2 < -10) = -10;
    
    y{2} = [v1./(1+exp(-v2)), 1]; % dim( 1 x P) P = nb hidden nodes

    vOut = y{2}*w{end}; % correspond à vk(n)
    vOut(vOut > 10) = 10;
    vOut(vOut < -10) = -10;
    
    Out = 1/(1+exp(-vOut)); % correspond à yk(n)
    err = T(k)-Out;
    
    %************************************
    % Back-Propagation
    %************************************
%     deltaOut = err*Out*(1-Out); % delta k (has this simple form because it is a sigmoid function)!!!
    delta{3} = err*Out*(1-Out); % delta k (has this simple form because it is a sigmoid function)!!!
    
    % Pas sûr pour ces deltas car je ne sais pas trop quel dimension ils
    % ont! en faut-il aussi pour le bias?
    delta{1} = (1./(1+exp(-v2))) * delta{3} .* w{3}(1:end-1)';  
    delta{2} = (v1.*exp(-v2))./(1+exp(-v2)).^2 * delta{3} .* w{3}(1:end-1)';
    
    
    
    %****** Weight Update *******
    prev_w{1} = Momentum*prev_w{1} + LRate*(y{1}(k,:)')*delta{1};
    w{1} = w{1} + prev_w{1};
    
    prev_w{2} = Momentum*prev_w{2} + LRate*(y{1}(k,:)')*delta{2};
    w{2} = w{2} + prev_w{2};
    
    prev_w{3} = Momentum*prev_w{3} + LRate*delta{3}*y{2}';
    w{3} = w{3} + prev_w{3};
    
    MSEs(k) = 0.5*err^2;

    end
    
    %Mean square error
    mse = (1/N)*sum(MSEs);
    ga(epoch+1) = mse;
    epoch = epoch + 1; 
    
    % permutation of the points
    permutation = randperm(N);
    y{1} = y{1}(permutation ,:);
    T = T(permutation);
    
    
end
plot(1:length(ga),ga);
epoch
weights = [w{1},w{2},w{end}];
    

end








