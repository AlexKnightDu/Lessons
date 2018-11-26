function [output_img] = scale(input_img, depth)
% This function is used to scale the unproper grey values into
% a proper range according to the depth of input img

% Get the maximum and minimum grey values
max_value = max(max(input_img));
min_value = min(min(input_img));

% Transform the negative value into the positive value
output_img = input_img - min_value;


% Scale the value into [0, 2^depth - 1]
output_img = (2 ^ depth - 1) / (max_value - min_value) * output_img;

% scale the image to [0, 2^L - 1] for better accuracy
laplacian_img = scale(laplacian_img, L);
sharpen_laplacian_img = scale(sharpen_laplacian_img, L);
sobel_gradient_img = scale(sobel_gradient_img, L);
smooth_5x5_sobel_img = scale(smooth_5x5_sobel_img, L);
mask_product_img = scale(mask_product_img, L);
sharpen_smooth_sobel_img = scale(sharpen_smooth_sobel_img, L);
power_law_tranform_img = scale(power_law_tranform_img, L);

end
