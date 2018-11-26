% Histogram of input images
function [ img_histogram ] = histogram(input_img,  depth)
% Used to generate the histogram of the input image
% Similiar with the project 1
input_img = uint8(input_img);
img_histogram = zeros(2^depth,1);
for i = 1:2^depth
    img_histogram(i) = length(find(input_img == i));
end

end  % function
