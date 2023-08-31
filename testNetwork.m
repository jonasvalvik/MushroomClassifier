function testNetwork(net, inputImg)

img = imread(inputImg);
resizedImg = imresize(img, [224, 224]);

[Label, Probability] = classify(net, resizedImg);

figure;
imshow(resizedImg);
title({"Type: " + char(Label), "Accuracy: " + num2str(max(Probability)*100, 6) + "%"})

end