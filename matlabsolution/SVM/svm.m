





function out = svm(X, tOut, eta, C,Kernel)


SV = size(X,1);
alpha = ones(SV,1)*0.01;
v = 0;
stop = false;
gamma = zeros(SV,1);
deltaAlpha = zeros(SV,1);

z = tOut*10;
epoch = 0;


while stop ~= true
    epoch = epoch + 1;
    for mu = 1 : SV % all pattern mu (SV)
   
        z(mu) = (alpha'.*tOut')*Kernel(:,mu) - v;

        gamma(mu) = tOut(mu)*z(mu);
        deltaAlpha(mu) = eta*(1-gamma(mu));
        alpha(mu) = alpha(mu) + deltaAlpha(mu);

        % Update alphas at boundaries
        if alpha(mu) <= 0
            alpha(mu) = 0.0;
        elseif alpha(mu) > C
            alpha(mu) = C;
        end
        
        zPlus = Inf;
        zMinus = -Inf;
       

        val = z(tOut==1 & alpha<C );
        if ~isempty(val)
            zPlus = min(val);
        else
            stop = true;
            fprintf('infinite value for zPlus');
            break
        end


        val2 = z(tOut==-1 & alpha<C );
        if ~isempty(val2)
            zMinus = max(val2);
        else
            stop = true;
            fprintf('infinite value for zMinus');
            break
        end
        u = 0.5*(zPlus + zMinus);
        v = u + v;
        z = z - u;
        
        %condition foireuse d'arrêt
%         zPlus - zMinus
        if abs(0.5* (zPlus - zMinus) - 1) < 0.1
            stop = true;
        end
    end
    if epoch == 3000
        stop = true;
    end
end
epoch
out = cell(1,2);
out{1} = alpha;
out{2} = v;

end