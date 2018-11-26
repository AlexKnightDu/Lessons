% Figures
function [] = figures( input_imgs, figure_title, image_titles, output_file)
% Show the pictures and the figures

figure('NumberTitle','off','Name', figure_title)
column = 2;
[temp, num] = size(input_imgs);
row = ceil(num / column);

for i = 1:row
  for j = 1:column
    if((i - 1)* column + j) <= num
      subplot(row, column, (i - 1)* column + j)
      imshow(uint8(input_imgs{(i - 1) * column + j}),[]);
      title(char(image_titles{(i - 1) * column + j}))
      if output_file
        imwrite(uint8(input_imgs{(i - 1) * column + j} * 255), [char(image_titles{(i - 1) * column + j}), '.jpg']);
      end
    end
  end
end

end
