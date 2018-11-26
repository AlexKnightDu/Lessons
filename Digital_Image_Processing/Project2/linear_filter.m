% Linear filter
function [output_img] = linear_filter(input_img, mask)
% This function is used to mask to input image

% Get the height and width of the mask
[m, n] = size(mask);

% Calculate the size of padding according to the size of mask
padding_m = floor((m - 1) / 2);
padding_n = floor((n - 1) / 2);
% Use zero padding to mask the boundary
padding_input_img = padarray(input_img, [padding_m, padding_n]);

% Get the height and width of the image
[M, N] = size(input_img);

% Initialize and calculate output image
output_img = zeros(M, N);
for i = padding_m + 1:padding_m + M
    for j = padding_n + 1:padding_n + N
        output_img(i - padding_m, j - padding_n) = sum(sum(padding_input_img(i-padding_m:i+padding_m,j-padding_n:j+padding_n) .* mask));
    end
end

end
