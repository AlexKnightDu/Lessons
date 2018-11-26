% Solution of project 4: Generating different types of noise and comparing different noise reduction methods
function [] = noises(img_location)

% Close all the former windows
close all;


%% %%%%%%%%%%%%%%%%%%%% The noise patterns part %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear the environment
clc
clear;

% Set the default location of input image file that needs to add noise
noise_img_location = '../../images/Fig0503 (original_pattern).tif';
input_noise_img = double(imread(noise_img_location));

% Get the relative information of the i
noise_img_info = imfinfo(noise_img_location);
M = noise_img_info.Height;
N = noise_img_info.Width;
% For this program must be general to allow any
% gray-level image, so it needs to get the depth
% of the image
L = noise_img_info.BitDepth;


% Generating the noise and add them to the image that needs to add noise
a = 0; b = 20;
uniform_noise_img = noise_generator(input_noise_img, 'uniform', L,a,b);
uniform_noise_histogram = histogram(uniform_noise_img, L);

a = 0; b = 100;
gaussian_noise_img = noise_generator(input_noise_img, 'gaussian', L,a,b);
gaussian_noise_histogram = histogram(gaussian_noise_img, L);

a = 0; b = 400;
rayleigh_noise_img = noise_generator(input_noise_img, 'rayleigh', L,a,b);
rayleigh_noise_histogram = histogram(rayleigh_noise_img, L);

a = 0.1; b = 0.1;
impulse_noise_img = noise_generator(input_noise_img, 'impulse', L,a,b);
impulse_noise_histogram = histogram(impulse_noise_img, L);

a = 0.1; b = 1;
exponential_noise_img = noise_generator(input_noise_img, 'exponential', L,a,b);
exponential_noise_histogram = histogram(exponential_noise_img, L);

a = 0.1; b = 5;
gamma_noise_img = noise_generator(input_noise_img, 'gamma', L,a,b);
gamma_noise_histogram = histogram(gamma_noise_img, L);

input_img_histogram = histogram(input_noise_img, L);

%%%%%%%%%%%%%%%%%%%%%% Show pictures & Draw graphs %%%%%%%%%%%%%%%%%%%%%%%%%%%%

noise_figure(input_noise_img, input_img_histogram, 'Input Image', L, true);
noise_figure(uniform_noise_img, uniform_noise_histogram, 'Uniform noise', L, true);
noise_figure(gaussian_noise_img, gaussian_noise_histogram, 'Gaussian noise', L, true);
noise_figure(rayleigh_noise_img, rayleigh_noise_histogram, 'Rayleigh noise', L, true);
noise_figure(impulse_noise_img, impulse_noise_histogram, 'Impulse noise', L, true);
noise_figure(exponential_noise_img, exponential_noise_histogram, 'Exponential noise', L, true);
noise_figure(gamma_noise_img, gamma_noise_histogram, 'Gamma noise', L, true);



% %% %%%%%%%%%%%%%%%%%%%% The reduction filter part %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear the environment
clc
clear;

% the image that that to show the result of noise reduction
reduction_img_location = '../../images/Circuit.tif';
input_reduction_img = double(imread(reduction_img_location));

% Get the relative information of the i
reduction_img_info = imfinfo(reduction_img_location);
M = reduction_img_info.Height;
N = reduction_img_info.Width;
% For this program must be general to allow any
% gray-level image, so it needs to get the depth
% of the image
L = reduction_img_info.BitDepth;

% For this part, it is asked to investigate the noise reduction results using
% different mean filters and order statistics filters as the textbook did at
% pages 322-329, so I implemented the process accodring to the book and show
% the results accodring to the figures in the book as following:

%%%%%%%%%% Mean filters %%%%%%%%%%%%%%%%

% FIGURE 5.7 - P324
% (a) Original input image.
% (b) Image corrupted by additive Gaussian noise.
% (c) Filtering with an arithmetic mean filter of size 3 * 3.
% (d) Filtering with a geometric mean filter of size 3 * 3.
a = 0; b = 400;
gaussian_noise_img_400 = noise_generator(input_reduction_img, 'gaussian', L,a,b);
arithmetic_mean_reduction = spatial_filter(gaussian_noise_img_400, 'arithmetic_mean', 3, 3, 0);
geometric_mean_reduction = spatial_filter(gaussian_noise_img_400, 'geometric_mean', 3, 3, 0);

noise_reduction_figure('Figure 5.7',...
{input_reduction_img, gaussian_noise_img_400, arithmetic_mean_reduction, geometric_mean_reduction},...
{'Input image', 'Gaussian noise', 'Arithmetic mean', 'Geometric mean'},...
true);


% FIGURE 5.8 - P325
% (a) Image corrupted by pepper noise with a probability of 0.1.
% (b) Image corrupted by salt noise with a probability of 0.1.
% (c) Filtering (a) with a 3 * 3 contraharmonic filter of order 1.5.
% (d) Filtering (b) with a 3 * 3 contraharmonic filter of order -1.5.
a = 0.1; b = 0;
pepper_noise_img = noise_generator(input_reduction_img, 'impulse', L,a,b);
a = 0; b = 0.1;
salt_noise_img = noise_generator(input_reduction_img, 'impulse', L,a,b);
contraharmonic_mean_reduction_pepper = spatial_filter(pepper_noise_img, 'contraharmonic_mean', 3, 3, 1.5);
contraharmonic_mean_reduction_salt = spatial_filter(salt_noise_img, 'contraharmonic_mean', 3, 3, -1.5);
noise_reduction_figure('Figure 5.8',...
{pepper_noise_img, salt_noise_img, contraharmonic_mean_reduction_pepper, contraharmonic_mean_reduction_salt},...
{'Pepper noise', 'Salt noise', 'Contraharmonic mean Q=1.5', 'Contraharmonic mean Q=-1.5'},...
true);


% FIGURE 5.9 - P326
% Results of selecting the wrong sign in contraharmonic filtering.
% (a) Filtering Fig. 5.8(a) with a contraharmonic filter of size 3 * 3 and Q = -1.5.
% (b) Filtering Fig. 5.8(b) with a contraharmonic filter of size 3 * 3 and Q = 1.5.
wrong_contraharmonic_mean_reduction_pepper = spatial_filter(pepper_noise_img, 'contraharmonic_mean', 3, 3, -1.5);
wrong_contraharmonic_mean_reduction_salt = spatial_filter(salt_noise_img, 'contraharmonic_mean', 3, 3, 1.5);
noise_reduction_figure('Figure 5.9',...
{wrong_contraharmonic_mean_reduction_pepper, wrong_contraharmonic_mean_reduction_salt},...
{'Wrong contraharmonic mean Q=-1.5', 'Wrong contraharmonic mean Q=1.5'},...
true);

%%%%%%%%%%% Order-statistic filters %%%%%%%%%%%%%%%%

% FIGURE 5.10 - P328
% (a) Image corrupted by salt-and-pepper noise with probabilities Pa=Pb=0.1.
% (b) Result of one pass with a median filter of size 3 * 3.
% (c) Result of processing (b) with this filter.
% (d) Result of processing (c) with the same filter.
a = 0.1; b = 0.1;
impulse_noise_img = noise_generator(input_reduction_img, 'impulse', L,a,b);
median_reduction_1times = spatial_filter(impulse_noise_img, 'median', 3, 3, 0);
median_reduction_2times = spatial_filter(median_reduction_1times, 'median', 3, 3, 0);
median_reduction_3times = spatial_filter(median_reduction_2times, 'median', 3, 3, 0);
noise_reduction_figure('Figure 5.10',...
{impulse_noise_img, median_reduction_1times, median_reduction_2times, median_reduction_3times},...
{'Pepper and salt noise', 'Median filter 1 times', 'Median filter 2 times', 'Median filter 3 times'},...
true);

% FIGURE 5.11 - P328
% (a) Result of filtering Fig. 5.8(a) with a max filter of size 3 * 3.
% (b) Result of filtering 5.8(b) with a min filter of the same size.
max_reduction = spatial_filter(pepper_noise_img, 'max', 3, 3, 0);
min_reduction = spatial_filter(salt_noise_img, 'min', 3, 3, 0);
noise_reduction_figure('Figure 5.11',...
{max_reduction, min_reduction},...
{'Max filter', 'Min filter'},...
true);
%
% FIGURE 5.12 - P329
% (a) Image corrupted by additive uniform noise.
% (b) Image additionally corrupted by additive salt-and-pepper noise. Image (b) filtered with a 5 * 5:
% (c) arithmetic mean filter;
% (d) geometric mean filter;
% (e) median filter;
% (f) alpha-trimmed mean filter with d = 5.
a = -20;  b = 20;
uniform_noise_img = noise_generator(input_reduction_img, 'uniform', L,a,b);
a = 0.1; b = 0.1;
impulse_uniform_noise_img = noise_generator(uniform_noise_img, 'impulse', L,a,b);
arithmetic_mean_reduction_5x5 = spatial_filter(impulse_uniform_noise_img, 'arithmetic_mean', 5, 5, 0);
geometric_mean_reduction_5x5 = spatial_filter(impulse_uniform_noise_img, 'geometric_mean', 5, 5, 0);
median_reduction_5x5 = spatial_filter(impulse_uniform_noise_img, 'median', 5, 5, 0);
alpha_trimmed_mean_reduction_5x5 = spatial_filter(impulse_uniform_noise_img, 'alpha_trimmed_mean', 5, 5, 5);
noise_reduction_figure('Figure 5.12',...
{uniform_noise_img, impulse_uniform_noise_img, arithmetic_mean_reduction_5x5, ...
geometric_mean_reduction_5x5, median_reduction_5x5, alpha_trimmed_mean_reduction_5x5},...
{'Uniform noise', 'Impulse noise added to uniform noise', 'Arithmetic mean 5x5', ...
'Geometric mean 5x5', 'Median filter 5x5', 'Alpha trimmed mean 5x5'},...
true);
