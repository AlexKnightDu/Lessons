% Solution of project 8: Morphological Processing
function [] = morphological_processing(img_location)

% Close all the other windows
close all;

%%%%%%%%%%%%%%%%%%%%%%%%% Opening by reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clear the environment
clc
clear;

% Set the default location of input image file
img_location = '../../images/Fig0929(a)(text_image).tif';
input_img = double(imread(img_location));

% Get the relative information of the image
info = imfinfo(img_location);
M = info.Height;
N = info.Width;

% Get the erosion image and dilation image
erosion_img = operations(input_img, 'erosion', 51,1);
dilation_img = operations(erosion_img, 'dilation', 51,1);

% First assign the dilation image to reconstruction image
% Then use dilation operation on reconstruction image
% Until the reconstruction image doesn't change any more
reconstruction_img = dilation_img;
while(true)
    new_img = operations(reconstruction_img, 'dilation', 3, 3);
    new_img = new_img & input_img;
    if all(all(new_img == reconstruction_img))
        break;
    end
    reconstruction_img = new_img;
end

% Show the pictures and output them
figures(...
{input_img, erosion_img, dilation_img, reconstruction_img},...
'FIG9.29 Opening by reconstruction',...
{'FIG9.29(a) Origin input image', 'FIG9.29(b) Erosion', 'FIG9.29(c) Opening', ...
'FIG9.29(d) Reconstruction'},...
true)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Filling holes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clear the environment
clc
clear;

% Set the default location of input image file
img_location = '../../images/Fig0929(a)(text_image).tif';
input_img = double(imread(img_location));

% Get the relative information of the image
info = imfinfo(img_location);
M = info.Height;
N = info.Width;

% Get the conplement of the input image
complement_img = ~input_img;

% Get the erosion image and dilation image
erosion_img = operations(complement_img, 'erosion', 51, 3);
dilation_img = operations(erosion_img, 'dilation', 51, 3);

% First assign the dilation image to filling image
% And add edges to it
filling_img = dilation_img;
filling_img([1, end], :) = 1;
filling_img(:, [1, end]) = 1;
filling_img = filling_img & complement_img;

marker_img = zeros(M,N);
marker_img([1, end], :) = input_img([1, end], :) == 1;
marker_img(:, [1, end]) = input_img(:, [1, end]) == 1;

% Then use dilation operation on filling image
% Until the filling image doesn't change any more
while(true)
    new_img = operations(filling_img, 'dilation', 3, 3);
    new_img = new_img & complement_img;
    if all(all(new_img == filling_img))
        break;
    end
    filling_img = new_img;
end

filling_img = ~filling_img;

% Show the pictures and output them
figures(...
{input_img, complement_img, marker_img, filling_img},...
'FIG9.31',...
{'FIG9.31(a) Origin input image', 'FIG9.31(b) Complement', 'FIG9.31(c) Marker',...
 'FIG9.31(d) Hole-filling'},...
true)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Border clearing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear the environment
clc
clear;

% Set the default location of input image file
img_location = '../../images/Fig0929(a)(text_image).tif';
input_img = double(imread(img_location));

% Get the relative information of the image
info = imfinfo(img_location);
M = info.Height;
N = info.Width;

% First get the border of the input image
border_img = zeros(M,N);
border_img([1, end], :) = input_img([1, end], :);
border_img(:, [1, end]) = input_img(:, [1, end]);

% Then use dilation operation on filling image
% Until the filling image doesn't change any more
while(true)
    new_img = operations(border_img, 'dilation', 3, 3);
    new_img = new_img & input_img;
    imshow(new_img);
    if all(all(new_img == border_img))
        break;
    end
    border_img = new_img;
end
border_clearing_img = input_img - border_img;

% Show the pictures and output them
figures(...
{border_img, border_clearing_img, input_img},...
'FIG9.32',...
{'FIG9.32(a) Marker', 'FIG9.32(b) Border clearing', 'FIG9.32 Origin input image'},...
true)


end
