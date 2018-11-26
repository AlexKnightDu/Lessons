% Noise generator
function [ output_img ] = noise_generator(input_img, pattern, depth, a, b)
% This function is used for Project4 originally, but it also can be used
% to generate the gaussian noise here in Project5
% Input:
%   pattern - the pattern of noise, it need to be the value of next several values:
%             {'uniform', 'gaussian', 'rayleigh', 'gamma', 'exponential', 'impulse'}
%             the erlang noise is same as gamma noise
%             the salt and pepper noise is same as impulse noise
%   a - the parameter needed in some noise pattern
%   b - the parameter needed in some noise pattern

% Define the noise patterns can be generated
patterns = {'uniform', 'gaussian', 'rayleigh', 'gamma', 'exponential', 'impulse'};

% Get the height and width of the input image
[M,N] = size(input_img);

% Initialize the noise and output image matrice
noise = zeros(M,N);
output_img = zeros(M,N);

% Generate the noise accodring to the pattern and the random matrice rand(M,N)
switch lower(pattern)
  case patterns{1}
    % The uniform noise
    noise = a + (b - a) * rand(M, N);
  case patterns{2}
    % The Gaussian noise
    noise = a + sqrt(b) * randn(M, N);
  case patterns{3}
    % The Rayleigh noise
    noise = a + sqrt(-b * log(1 - rand(M, N)));
  case patterns{4}
    % The Gamma or we say Erlang noise
    for i = 1:b
      noise = noise + (-1 / a) * log(1 - rand(M, N));
    end
  case patterns{5}
    % The Exponential noise
    noise = (-1 / a) * log(1 - rand(M, N));
  case patterns{6}
    % The impulse or we say Salt and Pepper noise
    random = rand(M, N);
    noise(random < a) = 1;
    noise(random >= a & random < (a + b)) = 2^depth - 1;
    output_img(random < a) = -input_img(random < a);
    output_img(random >= a & random < (a + b)) = -input_img(random >= a & random < (a + b));
  otherwise
    error(['The', ' ', pattern, ' ', 'noise pattern can not be found.'])
end
output_img = input_img + noise + output_img;

end
