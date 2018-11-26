% Solution of project 9: Image segmentation
function [] = image_segmentation(img_location)

% Clear the environment
close all;

%% %%%%%%%%%%%%%%%%%%%%%%% Edge detection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear;

% Set the default location of input image file
img_location = '../../images/building.tif';
input_img = double(imread(img_location));

% Get the relative information of the image
info = imfinfo(img_location);
M = info.Height;
N = info.Width;


%% Roberts edge detection
roberts_x_img = abs(filters(input_img, 'roberts', 'x'));
roberts_y_img = abs(filters(input_img, 'roberts', 'y'));
roberts_img = (roberts_x_img + roberts_y_img);

% Show the pictures and output them
figures(...
{input_img, roberts_x_img, roberts_y_img, roberts_img},...
'Roberts ',...
{'Origin input image', 'Roberts |gx|', 'Roberts |gy|', 'Roberts |gx|+|gy|'},...
true)


%% Prewitt edge detection
prewitt_x_img = abs(filters(input_img, 'prewitt', 'x'));
prewitt_y_img = abs(filters(input_img, 'prewitt', 'y'));
prewitt_img = (prewitt_x_img + prewitt_y_img);

% Show the pictures and output them
figures(...
{input_img, prewitt_x_img, prewitt_y_img, prewitt_img},...
'Prewitt ',...
{'Origin input image', 'Prewitt |gx|', 'Prewitt |gy|', 'Prewitt |gx|+|gy|'},...
true)


%% Sobel edge detection
sobel_x_img = abs(filters(input_img, 'sobel', 'x'));
sobel_y_img = abs(filters(input_img, 'sobel', 'y'));
sobel_img = (sobel_x_img + sobel_y_img);

% Show the pictures and output them
figures(...
{input_img, sobel_x_img, sobel_y_img, sobel_img},...
'Sobel ',...
{'Origin input image', 'Sobel |gx|', 'Sobel |gy|', 'Sobel |gx|+|gy|'},...
true)

%% Smooth the result of sobel
smooth_input_img = filters(input_img, 'smooth');
sobel_smooth5x5_x_img = abs(filters(smooth_input_img, 'sobel', 'x'));
sobel_smooth5x5_y_img = abs(filters(smooth_input_img, 'sobel', 'y'));
sobel_smooth5x5_img = (sobel_smooth5x5_x_img + sobel_smooth5x5_y_img);

% Show the pictures and output them
figures(...
{smooth_input_img, sobel_smooth5x5_x_img, sobel_smooth5x5_y_img, sobel_smooth5x5_img},...
'Sobel smoothed input image with 5x5 mask',...
{'Smoothed input image', 'Sobel Smoothed |gx|', 'Sobel Smoothed |gy|', 'Sobel Smoothed |gx|+|gy|'},...
true)


%% Marr-Hildreth edge detection.
% The Laplacian of Gaussian
LoG_img = filters(input_img, 'log', [25,4]);
sign = zeros(M, N);
sign(LoG_img > 0) = 1;
sign(LoG_img < 0) = -1;

% Find zero crossings with a threshold of 0.
zero_crossing_0 = zeros(M, N);
for i = 2:M-1
    for j = 2:N-1
        if sign(i-1, j-1) * sign(i+1, j+1) < 0 || ...
                sign(i-1, j) * sign(i+1, j) < 0 || ...
                sign(i, j-1) * sign(i, j+1) < 0 || ...
                sign(i+1, j-1) * sign(i-1, j+1) < 0
            zero_crossing_0(i, j) = 1;
        end
    end
end

% Find zeros crossings with threshold of 4%
zero_crossing_04 = zeros(M, N);
threshold = 0.04 * (max(LoG_img(:)) - min(LoG_img(:)));
for i = 2:M-1
    for j = 2:N-1
        if sign(i-1, j-1) * sign(i+1, j+1) < 0 && ...
                abs(LoG_img(i-1, j-1) - LoG_img(i+1, j+1)) > threshold
            zero_crossing_04(i, j) = 1;
        elseif sign(i, j-1) * sign(i, j+1) < 0 && ...
                abs(LoG_img(i, j-1) - LoG_img(i, j+1)) > threshold
            zero_crossing_04(i, j) = 1;
        elseif sign(i-1, j) * sign(i+1, j) < 0 && ...
                abs(LoG_img(i-1, j) - LoG_img(i+1, j)) > threshold
            zero_crossing_04(i,j) = 1;
        elseif sign(i+1, j-1) * sign(i-1, j+1) < 0 && ...
                abs(LoG_img(i+1, j-1) - LoG_img(i-1, j+1)) > threshold
            zero_crossing_04(i,j) = 1;
        end
    end
end

% To show the image seems like in the book, do some scaling
LoG_img = LoG_img - min(LoG_img(:));
output_LoG_img = LoG_img / max(LoG_img(:));
output_LoG_img = output_LoG_img * 255;
zero_crossing_0 = zero_crossing_0 * 255;
zero_crossing_04 = zero_crossing_04 * 255;

% Show the pictures and output them
figures(...
{input_img, output_LoG_img, zero_crossing_0, zero_crossing_04},...
'Marr-Hildreth edge detection',...
{'Origin input image', 'LoG ', 'Zero crossings with threshold 0', 'Zero crossings with threshold 0.04'},...
true)

%% Canny edge detection.
% Gaussian blur.
gaussian_img = filters(input_img, 'gaussian', [25, 4]);

% Calculate the gradient magnitude and angle images.
magnitude_x_img = filters(gaussian_img, 'sobel', 'x');
magnitude_y_img = filters(gaussian_img, 'sobel', 'y');
magnitude_img = sqrt(magnitude_x_img.^2 + magnitude_y_img.^2);

angle = zeros(M, N);
% For different direction set different number:
%   Horizonal edge - 1
%   Vertical edge - 2
%   -45 degree edge - 3
%   +45 degree edge - 4
for i = 1:M
    for j = 1:N
        if magnitude_y_img(i, j) == 0
            angle(i, j) = 1;
        else
            t = magnitude_x_img(i, j) / magnitude_y_img(i, j);
            if abs(t) >= tan(0.375 * pi)
                angle(i, j) = 1;
            elseif abs(t) <= tan(0.125 * pi)
                angle(i, j) = 2;
            elseif t > 0
                angle(i, j) = 3;
            else
                angle(i, j) = 4;
            end
        end
    end
end

% Apply nonmaxiMa suppression to the gradient magnitude image.
suppression_img = zeros(M, N);
for i = 2:M-1
    for j = 2:N-1
        t = angle(i, j);
        m = magnitude_img(i - 1:i + 1, j - 1:j + 1);
        if (t == 1 && m(2, 2) > m(3, 2) && m(2, 2) > m(1, 2)) || ...
                (t == 2 && m(2, 2) > m(2, 3) && m(2, 2) > m(2, 1)) || ...
                (t == 3 && m(2, 2) > m(1, 1) && m(2, 2) > m(3, 3)) || ...
                (t == 4 && m(2, 2) > m(1, 3) && m(2, 2) > m(3, 1))
            suppression_img(i,j) = m(2, 2);
        end
    end
end

% Scale the suppression image to [0,1]
suppression_img = suppression_img - min(suppression_img(:));
suppression_img = suppression_img / max(suppression_img(:));

% Use double thresholding and connectivity analysis to detect and link edges
low_threshold_img = zeros(M, N);
high_threshold_img = zeros(M, N);
low_threshold_img(suppression_img > 0.04) = 1;
high_threshold_img(suppression_img > 0.10) = 1;

% Use queue to store high pixel positions.
[x, y] = find(high_threshold_img);
head = 1;
tail = length(x);
canny_edge_img = high_threshold_img;
while(head <= tail)
    x0 = x(head);
    y0 = y(head);
    for i = x0 - 1:x0 + 1
        for j = y0 - 1:y0 + 1
            if low_threshold_img(i, j) && ~canny_edge_img(i, j)
                tail = tail + 1;
                x(tail) = i;
                y(tail) = j;
                canny_edge_img(i, j) = 1;
            end
        end
    end
    head = head + 1;
end

% To show the image seems like in the book, do some scaling
suppression_img = suppression_img * 255;
low_threshold_img = low_threshold_img * 255;
high_threshold_img = high_threshold_img * 255;
canny_edge_img = canny_edge_img * 255;


% Show the pictures and output them
figures(...
{input_img, gaussian_img, magnitude_img, suppression_img},...
'Canny edge detection',...
{'Origin input image', 'Gaussian blur', 'Gradient magnitude', 'Nonmaxima suppression'},...
true)
figures(...
{input_img, low_threshold_img, high_threshold_img, canny_edge_img},...
'Canny edge detection use double thresholding',...
{'Origin input image', 'Low threshold 0.04', 'High threshold 0.10', 'Canny edge detection'},...
true)


%% %%%%%%%%%%%%%%%%% Thresholding segmentation  %%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear;

% Set the default location of input image file
img_location = '../../images/polymersomes.tif';
input_img = double(imread(img_location));

% Get the relative information of the image
info = imfinfo(img_location);
M = info.Height;
N = info.Width;


% global thresholding method.
t = mean(mean(input_img));
delta = 0.1;
delta_t = 255;

while(delta_t > delta)
    delta_t = t;
    G1 = input_img(input_img > t);
    G2 = input_img(input_img <= t);
    mG1 = mean(G1);
    mG2 = mean(G2);
    t = (mG1 + mG2) / 2;
    delta_t = abs(delta_t - t);
end

global_threshold_img = zeros(M, N);
global_threshold_img(input_img > t) = 1;
global_threshold_img = global_threshold_img * 255;
figures(...
{input_img, global_threshold_img},...
'Global thresholding',...
{'Original input image', 'Global thresholding'},...
true)


% Otsu's method
L = 256;
Pi = zeros(L, 1);
P1k = zeros(L, 1);
mk = zeros(L, 1);

for i = 1:L
    Pi(i) = sum(sum(input_img == (i - 1))) / (M * N);
    if i == 1
        P1k(i) = Pi(i);
    else
        P1k(i) = P1k(i - 1) + Pi(i);
        mk(i) = mk(i - 1) + i * Pi(i);
    end
end
mG = mk(L);

sigma = ((mG * P1k - mk) .^ 2) ./ (P1k .* (1 - P1k));
sigma(isnan(sigma)) = 0;
k = mean(find(sigma == max(sigma))) - 1;

otsu_img = zeros(M, N);
otsu_img(input_img > k) = 1;
otsu_img = otsu_img * 255;
figures(...
{input_img, otsu_img},...
'Otsu`s method',...
{'Original input image', 'Otsu`s method'},...
true)

end
