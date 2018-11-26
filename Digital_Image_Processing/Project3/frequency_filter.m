% Frequency filters
function [ output_img ] = frequency_filter( input_img, pattern, pass_type, D0, n)
  % Input:
  %   input_img - the input image
  %   pattern - the pattern of filter, it need to be the value of next several values:
  %             {'ideal', 'butterWorth', 'gaussian'}
  %   pass_type - high pass or low pass
  %   D0 - the 'distance' D0
  %   n - the parameter used in the butterworth pattern
  % Output:
  %   the image after filtering


% The allowed patterns and high/low pass
patterns = {'ideal', 'butterworth', 'gaussian'};
pass_types = {'lowpass', 'highpass'};

% Convert the data type
input_img = double(input_img);

% Initialize
[M,N] = size(input_img);
H = zeros(M,N);


switch lower(char(pattern))
  case patterns{1}
    % Ideal 
    for i = 1:M
      for j = 1:N
        if sqrt((i - M/2) ^ 2 + (j - N/2) ^ 2) <= D0
          H(i,j) = 1;
        end
      end
    end
  case patterns{2}
    % Butterworth
    for i = 1:M
      for j = 1:N
        H(i,j) = 1 / (1 + (sqrt((i - M/2) ^ 2 + (j - N/2) ^ 2) / D0) ^ (2 * n));
      end
    end
  case patterns{3}
    % Gaussian
    for i = 1:M
      for j = 1:N
          H(i,j) = exp(-(sqrt((i - M/2) ^ 2 + (j - N/2) ^ 2) ^ 2) / (2*(D0^2)));
      end
    end
  otherwise
    error(['The', ' ', pattern, ' ', 'filter pattern can not be found.'])
end

switch lower(char(pass_type))
  case pass_types{1}
    H = H;
  case pass_types{2}
    H = 1 - H;
end

output_img = H .* input_img;
end
