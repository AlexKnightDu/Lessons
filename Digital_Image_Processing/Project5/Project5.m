% Solution of project 5: Image restoration
function [] = image_restoration(img_location)

% Clear the environment
close all;
clc;
clear;

% Set the default location of input image file
img_location = '../../images/book_cover.jpg';
input_img = double(imread(img_location));

% Get the relative information of the image
info = imfinfo(img_location);
M = info.Height;
N = info.Width;
depth = info.BitDepth;

% Blurring the input image
dft_input_img = fftshift(fft2(input_img));
blur_img = zeros(M, N);
for i = 1:M
    for j = 1:N
      % Use the H(u,v) to get the blurring with motion effects
        H = blurring(i - M/2, j - N/2, 0.1, 0.1, 1);
        blur_img(i,j) = dft_input_img(i,j) * H;
    end
end
blur_img = ifft2(ifftshift(blur_img));
% Show the pictures and output them
figures(...
{input_img, blur_img},...
'Bluring degradation with a=b=0.1 & T=1',...
{'Origin input image', 'Blurred image'},...
true)

% For different variance gaussian noises
for variance = [650, 65, 6.5]
  % Add gaussian noises to the blurred image
  gaussian_img = noise_generator(blur_img, 'gaussian', depth, 0, variance);

  % Restore the image using the inverse filter.
  dft_blur_img = fftshift(fft2(blur_img));
  direct_inverse_img = zeros(M, N);
  inverse_img = zeros(M, N);
  for i = 1:M
      for j = 1:N
          H = blurring(i - M/2, j - N/2, 0.1, 0.1, 1);
          direct_inverse_img(i,j) = (dft_blur_img(i,j) / H);
          % For there with be conditions H = 0
          % we need to set a threshold such as 0.001
          if abs(H) < 0.001
              H = 0.001;
          end
          inverse_img(i,j) = (dft_blur_img(i,j) / H);
      end
  end
  direct_inverse_img = uint8(real(ifft2(ifftshift(direct_inverse_img))));
  inverse_img = uint8(real(ifft2(ifftshift(inverse_img))));

  % Restore the image using wiener deconvolution filters.
  dft_gaussian_img = fftshift(fft2(gaussian_img));
  wiener_img = zeros(M, N);
  for i = 1:M
      for j = 1:N
          H = blurring(i - M/2, j - N/2, 0.1, 0.1, 1);
          wiener_img(i,j) = (dft_gaussian_img (i,j) / H) * (abs(H)^2 / (abs(H)^2 + 0.01));
      end
  end
  wiener_img = uint8(real(ifft2(ifftshift(wiener_img))));

  % Show the pictures and output them
  figures(...
  {gaussian_img, direct_inverse_img, inverse_img, wiener_img},...
  ['Restore the image with gaussian noise(var=', int2str(variance), ')'],...
  {['Gaussian noise image(var=', int2str(variance), ')'], ['Direct inverse restore (var=',...
   int2str(variance), ')'], ['Inverse restore (var=', int2str(variance), ')'], ['Wiener restore K = 0.01(var=', int2str(variance), ')']},...
  true)
end


% Investigate the performance of wiener filter
% Add gaussian noises to the blurred image
gaussian_img = noise_generator(blur_img, 'gaussian', depth, 0, 65);

for K = [0.1, 0.01, 0.001, 0.0001]
  % Restore the image using wiener deconvolution filters.
  dft_gaussian_img = fftshift(fft2(gaussian_img));
  wiener_img = zeros(M, N);
  for i = 1:M
      for j = 1:N
          H = blurring(i - M/2, j - N/2, 0.1, 0.1, 1);
          wiener_img(i,j) = (dft_gaussian_img (i,j) / H) * (abs(H)^2 / (abs(H)^2 + K));
      end
  end
  wiener_img = uint8(real(ifft2(ifftshift(wiener_img))));

  % Show the pictures and output them
  figures({wiener_img}, ['Restore the image using wiener filter(K=', num2str(K), ')'],...
  {['Wiener filter(K=', num2str(K), ')']}, true)
end
end
