% Solution of project 6: Geometric transform
function [] = geometric_transform(img_location)

% Clear the environment
close all;
clc
clear;

% Set the default location of input image file
img_location = '../../images/ray_trace_bottle.tif';
input_img = double(imread(img_location));

% Get the relative information of the i
info = imfinfo(img_location);
M = info.Height;
N = info.Width;

% Define the spatial transformation and interpolation patterns
transform_patterns = { 'translate', 'rotate', 'scale' };
interpolate_patterns = {'nearest', 'bilinear'};

% Get all the parameters accodring to the order of the patterns
all_parameters = {...
{[200,300],[-200,300]},...
{[90,0],[30,0],[45,0]},...
{[1,1.2],[2,1],[3,3]}...
};


% All three spatial transformation patterns and two interpolation patterns
% Get the output image and resize the output image to avoid cropping
for i = 1:3
  transform_pattern = transform_patterns{i};
  parameters = all_parameters{i};
  for parameter = parameters
      nearest_img = spatial_transform(input_img, transform_pattern, 'nearest', false, parameter{1});
      resize_nearest_img = spatial_transform(input_img, transform_pattern, 'nearest', true, parameter{1});
      bilinear_img = spatial_transform(input_img, transform_pattern, 'bilinear', false, parameter{1});
      resize_bilinear_img = spatial_transform(input_img, transform_pattern, 'bilinear', true, parameter{1});

      %%%%%%%%%%%%%%%%%%%%%%%%%% Show pictures %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      figure('NumberTitle', 'off', 'Name', ['The ', ' ', char(transform_pattern), ' transform'])
      if strcmp(char(transform_pattern), 'rotate')
        img_title = [int2str(parameter{1}(1)), ' degree'];
      else
        img_title = [int2str(parameter{1}(1)), ',', int2str(parameter{1}(2))];
      end
      subplot(2,2,1);
      imshow(nearest_img);
      title(['nearest - ', img_title]);
      subplot(2,2,2);
      imshow(resize_nearest_img);
      title('After resize');
      subplot(2,2,3);
      imshow(bilinear_img);
      title(['bilinear - ', img_title]);
      subplot(2,2,4);
      imshow(resize_bilinear_img);
      title('After resize');

      %%%%%%%%%%%%%%%%%%%%%%%%%% Output the file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      imwrite(nearest_img, ['nearest - ', img_title, '.jpg']);
      imwrite(resize_nearest_img, ['resize nearest - ', img_title, '.jpg']);
      imwrite(bilinear_img, ['bilinear - ', img_title, '.jpg']);
      imwrite(resize_bilinear_img, ['resize bilinear - ', img_title, '.jpg']);
  end
end
end
