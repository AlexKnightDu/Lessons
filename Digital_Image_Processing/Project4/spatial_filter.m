%% Spatial filter used for reducing the noises
function [ output_img ] = spatial_filter(input_img, pattern, m, n, parameter)
% Input:
%   input_img - the image needed to reduce the noises
%   pattern - the pattern of spatial filters, it need to be the value of next
%   several values:
%         mean filters
%             {'arithmetic_mean', 'geometric_mean', 'harmonic_mean', 'contraharmonic_mean',
%              'alpha_trimmed_mean'}
%         order statistics filters
%             {'median', 'max', 'min', 'midpoint'}
%
%   m - the height of the mask
%   n - the width of the mask
%   parameter - the parameter needed in some spatial filter
% Output:
%   the noise-reduced image

% Calculate the size of padding according to the size of mask
padding_m = floor((m - 1) / 2);
padding_n = floor((n - 1) / 2);

% Use zero padding to mask the boundary
padding_input_img = padarray(input_img, [padding_m, padding_n]);

% Get the height and width of the image
[M, N] = size(input_img);

% Initialize and calculate output image
output_img = zeros(M, N);
for i = padding_m + 1:padding_m + M
    for j = padding_n + 1:padding_n + N
        img_in_mask = padding_input_img(i - padding_m:i + padding_m, j - padding_n:j+padding_n);
        output_img(i - padding_m, j - padding_n) = mask(img_in_mask, pattern, m, n, parameter);
    end
end
end


%% Mask operation for the image area needed to be masked
function [ mask_result ] = mask(img_in_mask, pattern, m, n, parameter)
% Actually this function is just take the switch codes out
% of the above function, it looks like more clear in this
% way and easy to modify.

% Define the spatial filter patterns can be used
patterns = {'arithmetic_mean', 'geometric_mean', 'harmonic_mean', 'contraharmonic_mean',...
   'alpha_trimmed_mean', 'median', 'max', 'min', 'midpoint'};

switch lower(pattern)
    case patterns{1}
        mask_result = sum(sum(img_in_mask)) / (m * n);
    case patterns{2}
        mask_result = prod(prod(img_in_mask)) ^ (1 / (m * n));
    case patterns{3}
        mask_result = (m * n) / (sum(sum(1 ./ img_in_mask)));
    case patterns{4}
        % mask_result = sum(img_in_mask .^ (parameter + 1)) / sum(img_in_mask .^ parameter);
        mask_result = sum(sum(img_in_mask .^ (parameter + 1))) / sum(sum(img_in_mask .^ parameter));
    case patterns{5}
        sorted_pixels = sort(img_in_mask);
        low = round(parameter / 2);
        high = parameter - low;
        sorted_pixels = sorted_pixels(low : m * n - high - 1);
        mask_result = sum(sum(sorted_pixels)) / (m * n - parameter);
    case patterns{6}
        mask_result = median(median(img_in_mask));
    case patterns{7}
        mask_result = max(max(img_in_mask));
    case patterns{8}
        mask_result = min(min(img_in_mask));
    case patterns{9}
        mask_result = (max(max(img_in_mask)) + min(min(img_in_mask))) / 2;
    otherwise
        error(['The', ' ', pattern, ' ', 'filter can not be found.'])
end
end

end
