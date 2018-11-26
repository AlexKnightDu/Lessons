% Solution of project 1: Histogram Equalization
function[] = histogram_equalization(img_location)

% Clear the environment
close all;
clc;
clear;

% Set the default location of input image file
img_location = '../images/Fig2.jpg';
input_img = imread(img_location);

% Get the relative information of the i
info = imfinfo(img_location);
MN = info.Width * info.Height;

% For this program must be general to allow any
% gray-level image, so it needs to get the depth
% of the image
L = info.BitDepth;

% Initialize the gray distribution and output image matrix
input_gray_distribution = zeros(2^L,1);
output_img = zeros(info.Height,info.Width);
output_gray_distribution = zeros(2^L,1);
transform = zeros(2^L,1);

% Calculate the gray distribution of input image
for i = 1:2^L
    input_gray_distribution(i) = length(find(input_img == (i - 1)));
end

% Get the pdf
input_img_pdf = input_gray_distribution / MN;

% Calculate the transformation
for i = 1:2^L
  transform(i) = (2^L - 1) * sum(input_img_pdf(1:i));
end

% Calculate the output image
for i = 1:info.Width
    for j = 1:info.Height
        output_img(j,i) = transform(double(input_img(j,i)) + 1);
    end
end

% Format the values of pixels
output_img = uint8(output_img);

% Calculate the gray distribution of output image
for i = 1:2^L
    output_gray_distribution(i) = length(find(output_img == i));
end


%%%%%%%%%%%%%%% Show pictures & Draw graphs %%%%%%%%%%%%%%%%%
% Show pictures and graphs seperately
figure('NumberTitle','off','Name','Histogram Of Input Image')
bar(input_gray_distribution,0.7);
xlabel('Gray Level','FontSize',16);
ylabel('Distribution','FontSize',16);
set(gca, 'XLim',[0 2^L]);
set(gca, 'YLim',[0 16000]);

figure('NumberTitle','off','Name','Histogram Of Output Image')
bar(output_gray_distribution,0.7);
xlabel('Gray Level','FontSize',16);
ylabel('Distribution','FontSize',16);
set(gca, 'XLim',[0 2^L]);
set(gca, 'YLim',[0 16000]);

figure('NumberTitle', 'off', 'Name', 'Transformation function')
plot(1:2^L, transform)
xlabel('Original Gray','FontSize',16);
ylabel('Transformed Gray','FontSize',16);
set(gca, 'XLim',[0 2^L]);
set(gca, 'YLim',[0 2^L]);

figure('NumberTitle','off','Name','Input Image')
imshow(input_img);

figure('NumberTitle','off','Name','Output Image')
imshow(output_img);

% Show them all together to compare
figure('NumberTitle','off','Name','Compare')
subplot(2,3,1);
imshow(input_img);
title('The Input Image');

subplot(2,3,2);
bar(input_gray_distribution,0.7);
title('Histogram Of Input Image');
xlabel('Gray Level','FontSize',16);
ylabel('Distribution','FontSize',16);
set(gca, 'XLim',[0 2^L]);
set(gca, 'YLim',[0 16000]);

subplot(2,3,3);
plot(1:2^L, transform)
title('The Transformation Function')
xlabel('Original Gray','FontSize',16);
ylabel('Transformed Gray','FontSize',16);
set(gca, 'XLim',[0 2^L]);
set(gca, 'YLim',[0 2^L]);

subplot(2,3,4);
imshow(output_img);
title('The Output Image');

subplot(2,3,5);
bar(output_gray_distribution,0.7);
title('Histogram Of Output Image');
xlabel('Gray Level','FontSize',16);
ylabel('Distribution','FontSize',16);
set(gca, 'XLim',[0 2^L]);
set(gca, 'YLim',[0 16000]);
