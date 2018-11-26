% Blurring degradation function
function [ output ] = blurring(u, v, a, b, T)
% blurring degradation function.

t = u * a + v * b;
output = (T / (pi * t)) * sin(pi * t) * exp(-1i * pi * t);

% when the output gets infinity or NaN, we just do nothing
output(isinf(output)) = 1;
output(isnan(output)) = 1;

end
