% Spatial transformation
function [output_img] = spatial_transform(input_img, transform_pattern, interpolate_pattern, resize, parameters)
% Input:
%   transform_pattern - the pattern of transformation, it need to be the value of next several values:
%                       { 'translate', 'rotate', 'scale'}
%   interpolate_pattern - the pattern of interpolation, it need to be the value of next several values:
%                         { 'nearest', 'bilinear' }
%   parameters - the parameters needed in the transformations
%   resize - resize to the image to avoid the crop after transformation



% Define the transformation patterns
transform_patterns =  { 'translate', 'rotate', 'scale' };

% Get the height and width of the input image
[M,N] = size(input_img);

H = M;
W = N;

% Transfrom the type of input image to be double
input_img = double(input_img);

switch lower(char(transform_pattern))
  case transform_patterns{1}
    % Translate
    T = [1, 0, 0;
        0, 1, 0;
        parameters(1), parameters(2), 1];

    % Calculate inverse matrix of T.
    T = inv(T);

    % Resize the image to avoid the crop
    if resize
        H = M + abs(parameters(1));
        W = N + abs(parameters(2));
    end

  case transform_patterns{2}
    % Rotate

    parameter = (parameters(1) / 180) * pi;
    % For positive x-axis points down in matlab coordinates, and rotating
    % from x to y means counter clockwise, so there needs to inverse T.
    T = [cos(parameter), sin(parameter), 0;
        -sin(parameter), cos(parameter), 0;
        0, 0, 1];

    % Resize the image to avoid the crop
    if resize
        H = ceil(sqrt(M^2 + N^2) + 2);
        W = H;
    end
    parameter(1) = 0;
    parameter(2) = 0;

  case transform_patterns{3}
    % Scale

    T = [parameters(1), 0, 0;
        0, parameters(2), 0;
        0, 0, 1];

    T = inv(T);

    % Resize the image to avoid the crop
    if resize
        H = ceil(M * parameters(1));
        W = ceil(N * parameters(2));
    end

    parameters(1) = 0;
    parameters(2) = 0;

  otherwise
    error(['The', ' ', transform_pattern, ' ', 'noise pattern can not be found.'])

end

% Get the output image by the matrix T
output_img = zeros(H, W);
for i = -H/2 + 1:H/2
  for j = -W/2 + 1:W/2
    p = [i j 1] * T;
    % Add the offset
    u = p(1) + (M + parameters(1))/2;
    v = p(2) + (N + parameters(2))/2;
    if u >= 1 && u <= M && v >= 1 && v <= N
        output_img(i + H/2, j + W/2) = interpolation(input_img, interpolate_pattern, u, v);
    end
  end
end

% Transfrom the data type to be uint8
output_img = uint8(output_img);

% Cut off invaild black border.
if (strcmp(char(transform_pattern), 'rotate') && resize)
    fy = find(max(output_img));
    fx = find(max(output_img, [], 2));
    output_img = output_img(min(fx):max(fx), min(fy):max(fy));
end

end
