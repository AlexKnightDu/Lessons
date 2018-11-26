% Edge detection filters
function [ output_img ] = filters(input_img, pattern, parameters)
% These filters used to implement the mask in the process of edge Detection.
% Especially basic edge detection.
%
% Input:
%   pattern - the pattern of filters, it need to be the value of next several values:
%             {'roberts', 'prewitt', 'sobel', 'smooth', 'gaussian', 'log'}
%   parameters - the parameters used in the different process of edge detection
%             In 'roberts' & 'prewitt' & 'sobel':
%                     only one parameter to inform the direction
%             In 'gussian' & 'log':
%                     the first parameter is filter size, and the second is
%                     standard deviation

patterns = {'roberts', 'prewitt', 'sobel', 'smooth', 'gaussian', 'log'};

% Get the mask
switch lower(pattern)
case patterns{1}
    mask = [-1, 0; 0, 1];
    if strcmp(parameters, 'y')
        mask = [0, -1; 1, 0];
    end
  case patterns{2}
    mask = [-1, -1, -1; 0, 0, 0; 1, 1, 1];
    if strcmp(parameters, 'y')
        mask = mask';
    end
  case patterns{3}
    mask = [-1, -2, -1; 0, 0, 0; 1, 2, 1];
    if strcmp(parameters, 'y')
        mask = mask';
    end
  case patterns{4}
    mask = ones(5,5) / 25;
  case patterns{5}
    r = (parameters(1) - 1)/2;
    [x, y] = meshgrid(-r:r, -r:r);
    mask = exp(-(x .* x + y .* y)/(2 * parameters(2) * parameters(2)));
    if sum(mask(:)) ~= 0
        mask  = mask/sum(mask(:));
    end
  case patterns{6}
    r = (parameters(1) - 1)/2;
    [x, y] = meshgrid(-r:r, -r:r);
    mask = exp(-(x .* x + y .* y)/(2 * parameters(2)^2));
    mask = mask .* (x .* x + y .* y - 2 * parameters(2)^2)/(parameters(2)^4);
    mask = mask - sum(mask(:))/numel(mask);
  otherwise
    error(['The', ' ', pattern, ' ', 'filter pattern can not be found.'])
end

[M,N] = size(input_img);
[m,n] = size(mask);

% Zero padding
padding_m = floor((m - 1)/2);
padding_m_ = m - padding_m - 1;
padding_n = floor((n - 1)/2);
padding_n_ = n - padding_n - 1;

padding_img = padarray(input_img, [padding_m, padding_n]);
padding_img = zeros(M + m - 1, N + m - 1);
padding_img(padding_m + 1:padding_m + M, padding_n + 1:padding_n + N) = input_img;

output_img = zeros(M, N);
for i = padding_m + 1:padding_m + M
    for j = padding_n + 1:padding_n + N
        temp = padding_img(i - padding_m:i + padding_m_, j - padding_n:j + padding_n_) .* mask;
        output_img(i - padding_m, j - padding_n) = sum(temp(:));
    end
end

end
