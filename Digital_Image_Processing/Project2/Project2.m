% Solution of project 2: Combining spatial enhancement methods
function[] = SpatialEnhancement(img_location)

% Clear the environment
close all;
clc
clear;

% Set the default location of input image file
img_location = '../images/skeleton_orig.tif';
input_img = double(imread(img_location));

% For this program must be general to allow any
% gray-level image, so it needs to get the depth
% of the image
info = imfinfo(img_location);
L = info.BitDepth;

% Set the laplacian and sobel gradient masks also smooth mask
laplacian_mask = [0,-1,0; -1,4,-1; 0,-1,0];
sobel_gradient_x_mask = [-1,-2,-1; 0,0,0; 1,2,1];
sobel_gradient_y_mask = [-1,0,1; -2,0,2; -1,0,1];
smooth_5x5_mask = ones(5,5) * 1/(5*5);


% Laplacian of the input image
laplacian_img = linear_filter(input_img, laplacian_mask);

% Use laplacian to sharpen the image
sharpen_laplacian_img = input_img + laplacian_img;

% Sobel gradient of the input image of direction x and y
sobel_gradient_x_img = linear_filter(input_img, sobel_gradient_x_mask);
sobel_gradient_y_img = linear_filter(input_img, sobel_gradient_y_mask);

% Sobel gradient of the input image it can also use square sum
sobel_gradient_img = abs(sobel_gradient_x_img) + abs(sobel_gradient_y_img);

% Use 5x5 mask to smooth
smooth_5x5_sobel_img = linear_filter(sobel_gradient_img, smooth_5x5_mask);

% Calculate the product mask of laplacian and smooth sobel
mask_product_img = sharpen_laplacian_img .* smooth_5x5_sobel_img / (2^L - 1);
sharpen_smooth_sobel_img = input_img + mask_product_img;

% Calculate the power-law transformation of smooth sobel image
power_law_tranform_img = (sharpen_smooth_sobel_img * (2^L - 1)).^ 0.5;

% Increase the gray level to handle the negative gray level
laplacian_img = laplacian_img - min(laplacian_img(:));
sharpen_laplacian_img =  sharpen_laplacian_img - min(sharpen_laplacian_img(:));
sobel_gradient_img = sobel_gradient_img - min(sobel_gradient_img(:));


%%%%%%%%%%%%%%%%%%%%%%%% Show pictures %%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('NumberTitle','off','Name','Combining spatial enhancement methods')
subplot(2,4,1);imshow(input_img,[0,2^L-1]);title('(a) Input image');
subplot(2,4,2);imshow(laplacian_img,[0,2^L-1]);title('(b) Laplacian of (a)');
subplot(2,4,3);imshow(sharpen_laplacian_img,[0,2^L-1]);title('(c) Sum of (a) and (b)');
subplot(2,4,4);imshow(sobel_gradient_img,[0,2^L-1]);title('(d) Sobel of (a)');
subplot(2,4,5);imshow(smooth_5x5_sobel_img,[0,2^L-1]);title('(e) Smooth (d) with a 5x5 average filter');
subplot(2,4,6);imshow(mask_product_img,[0,2^L-1]);title('(f) Product of (c) and (e)');
subplot(2,4,7);imshow(sharpen_smooth_sobel_img,[0,2^L-1]);title('(g) Sum of (a) and (f)');
subplot(2,4,8);imshow(power_law_tranform_img,[0,2^L-1]);title('(h) Gamma transformation result of (g)');

%%%%%%%%%%%%%%%%%%%%%%%% Output pictures %%%%%%%%%%%%%%%%%%%%%%%%%%%%
imwrite(uint8(input_img), '1_origin_image.jpg');
imwrite(uint8(laplacian_img), '2_laplacian_of_image.jpg')
imwrite(uint8(sharpen_laplacian_img), '3_sharpen_laplacian_image.jpg')
imwrite(uint8(sobel_gradient_img), '4_sobel_gradient_image.jpg')
imwrite(uint8(smooth_5x5_sobel_img), '5_smooth_sobel_image.jpg')
imwrite(uint8(mask_product_img), '6_product_laplacian_sobel.jpg')
imwrite(uint8(sharpen_smooth_sobel_img), '7_sharpen_smooth_sobel.jpg')
imwrite(uint8(power_law_tranform_img), '8_power_law_transform.jpg')
