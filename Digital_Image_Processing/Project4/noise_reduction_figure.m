% Show the noise images before and after spatial filter
function [] = noise_reduction_figure(figure_title, images, titles, output_file)
% If needs output the image file, set the output_file to be True

[temp, num] = size(images);

column = 2;
row = floor((num + 1)/ column);

figure('NumberTitle','off','Name',figure_title)
for i = 0:row-1
    for j = 1:column
        if (i * column + j <= num)
            subplot(row, column, i * column + j );
            imshow(uint8(images{i * column + j}));
            title(titles{i * column + j});
            if (output_file == true)
              imwrite(uint8(images{i * column + j}), [figure_title, '-', titles{i * column + j}, '.jpg']);
            end
        end
    end
end
if (output_file == true)
  saveas(gcf, [figure_title, '.jpg'])
end
end
