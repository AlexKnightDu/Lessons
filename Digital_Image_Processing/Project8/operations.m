% Erosion or dialation
function [output_img] = operations(input_img, operation, m, n)
% Process the input image with erosion or dialation
% Input:
%   operation - the operation needed, it need to be the value of next two values:
%                {'erosion', 'dilation'}
%   m - mask height, must be odd
%   n - mask width, must be odd


[M,N] = size(input_img);
H = (m - 1)/2;
W = (n - 1)/2;

% Zero padding
im_padded = zeros(M + m - 1, N + n - 1);
im_padded(H + 1:H + M,W + 1:W + N) = input_img;

output_img = zeros(M, N) == 1;
switch lower(operation)
    case 'erosion'
      for i = H + 1:H + M
        for j = W + 1:W + N
          % Considering the efficiency, use the square mask with 1s
          if all(all(im_padded(i - H:i + H,j - W:j + W)))
              output_img(i - H, j - W) = 1;
          end
        end
      end
    case 'dilation'
      for i = H + 1:H + M
        for j = W + 1:W + N
          if any(any(im_padded(i - H:i + H, j - W:j + W)))
              output_img(i - H, j - W) = 1;
          end
        end
      end
    otherwise
      error(['The', ' ', operation, ' ', 'operation pattern can not be found.'])
end
end
