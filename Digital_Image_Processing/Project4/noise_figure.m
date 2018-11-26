% Show the noise image and draw its histogram graph
function [] = noise_figure( input_img, histogram, img_title, depth, output_file)
% If needs output the image file, set the output_file to be True

figure('NumberTitle', 'off', 'Name', ['The ', ' ', lower(img_title)])
subplot(1,2,1)
imshow(uint8(input_img));
title(img_title)
subplot(1,2,2)
bar(histogram, 0.7);
xlabel('Gray Level','FontSize',16);
ylabel('Distribution','FontSize',16);
axis square
set(gca, 'XLim',[0 2^depth]);
set(gca, 'YLim',[0 3000]);
if (output_file == true)
  saveas(gcf, [img_title, '.jpg'])
end
end
