function err = fastForward(w,v,y)

y{1} = [1 1 1];
v{1} =  y{1}*w{1}; % correspond à vj(n)
v{2} =  y{1}*w{2}; % correspond à vj(n)

v{2}(v{2} > 10) = 10; % avoid the too large or too small exponential
v{2}(v{2} < -10) = -10;

y{2} = [v{1}./(1+exp(-v{2})), 1]; % dim( 1 x P) P = nb hidden nodes

v{3} = y{2}*w{end}; % correspond à vk(n)
v{3}(v{3} > 10) = 10;
v{3}(v{3} < -10) = -10;

y{end} = 1./(1+exp(-v{3})); % correspond à yk(n)
err = [1 1] - y{end};
end