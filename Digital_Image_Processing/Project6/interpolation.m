% Get the interpolation of points
function [output] = interpolation(input_img, pattern, x, y)
% Input:
%   pattern - the pattern of interpolation, it need to be the value of next two values:
%             {'nearest', 'bilinear'}
%   x - the x coordinates
%   y - the y coordinates


% Define the interpolation patterns
patterns = {'nearest', 'bilinear'};

% Get the height and width of the input image
[M,N] = size(input_img);

% Get the interpolation result accodring to the pattern
switch lower(char(pattern))
  case patterns{1}
    % nearest
    % There use the round(x), round(y) to find the nearest point
    output = input_img(round(x), round(y));
  case patterns{2}
    % bilinear
    xm = floor(x);
    ym = floor(y);
    if x == xm && y == ym
        output = input_img(x, y);
    elseif x == xm && y ~= ym
        output = (y - ym) * input_img(x, ym + 1) + (ym + 1 - y) * input_img(x, ym);
    elseif x ~= xm && y == ym
        output = (x - xm) * input_img(xm + 1, y) + (xm + 1 - x) * input_img(xm, y);
    else
        t1 = (x - xm) * input_img(xm + 1, ym) + (xm + 1 - x) * input_img(xm, ym);
        t2 = (x - xm) * input_img(xm + 1, ym + 1) + (xm + 1 - x) * input_img(xm, ym + 1);
        output = (y - ym) * t2 + (ym + 1 - y) * t1;
    end
  otherwise
    error(['The', ' ', pattern, ' ', 'interpolation pattern can not be found.'])
end

% Transform the data type to uint8
output = uint8(output);
end
