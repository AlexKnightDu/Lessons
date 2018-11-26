% Figures
function [] = frequency_filter_figures( input_imgs, image_info, cutoffs, output_file )
% Show the pictures and the figures

figure('NumberTitle','off','Name', ['The', ' ',char(image_info{1}), ' ',...
                                    char(image_info{2}), ' ', 'filter'])
column = 2;
[temp, num] = size(input_imgs);
row = ceil(num / column);

% Show the pictures in 2 pictures/lines format
% If the output_file flag is set, then output to files
for i = 1:row
  for j = 1:column
    if((i - 1)* column + j) <= num
      subplot(row, column, (i - 1)* column + j)
      imshow(uint8((input_imgs{(i - 1) * column + j})),[]);
      if (i - 1)* column + j == 1
        title('Origin input image')
      else
        title(['Cutoff = ', int2str(cutoffs((i - 1) * column + j))]);
      if output_file
        if (i - 1)* column + j > 1
          imwrite(uint8(input_imgs{(i - 1) * column + j}), [char(image_info{1}),...
           '_',char(image_info{2}), '_', int2str(cutoffs((i - 1) * column + j)), '.jpg']);
        end
      end
    end
  end
end

if output_file
  % Save the whole figure containing six pictures
  saveas(gcf, [char(image_info{1}), '_', char(image_info{2}), '_', 'filter', '.jpg'])
end
end
