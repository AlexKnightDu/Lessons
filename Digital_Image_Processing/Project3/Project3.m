% Solution of project 3: Filtering in frequency domain
function [] = frequency_filtering(img_location)

% Clear the environment
close all;
clc
clear;

% Set the default location of input image file
img_location = '../../images/characters_test_pattern.tif';
input_img = double(imread(img_location));

% Get the relative information of the image
info = imfinfo(img_location);
M = info.Height;
N = info.Width;

% Set the tested cutoff values, noise patterns and pass type
cutoffs = [10, 30, 60, 160, 460];
patterns = {'ideal', 'butterworth', 'gaussian'};
pass_types = {'lowpass', 'highpass'};


%  Get the Discrete Fourier Transform
dft = fft2(input_img, 2*M, 2*N);
dft = fftshift(dft);

% Test each pattern high/low pass with different cutoffs
for pass_type = pass_types
  for pattern = patterns
    % Collect the generated figures to show together later
    figures = {};
    figures{1} = input_img;

    for i = 1:numel(cutoffs)
      % Frequency filter
      filter_dft = frequency_filter(dft, pattern, pass_type, cutoffs(i), 2);
      filter_img = real(ifft2(ifftshift(filter_dft)));

      % For we use bigger dft image so we need to crop the image converted back
      filter_img = filter_img(1:M,1:N);
      figures{i+1} = filter_img;
    end

    % Show and output the generated figures together
    frequency_filter_figures(figures, {pattern, pass_type}, [0,cutoffs], true)
  end
end
end
