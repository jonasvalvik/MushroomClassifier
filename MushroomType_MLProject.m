%Importing data. Folder names are used as labels.
dataset = imageDatastore('Mushrooms_dataset', 'IncludeSubfolders', true,  'LabelSource','foldernames');

%Splitting dataset into training and validation data. 70% used for
%training.
[Training_Dataset, Validation_Dataset] = splitEachLabel (dataset, 0.7); 

%Loading GoogLeNet. 'analyzeNetwork' used to visualize the layers of it.
net = googlenet; 
%analyzeNetwork(net);  

%Storing allowed input size and resizing data images to 224x224 pixels.
Input_Layer_Size = net.Layers(1).InputSize(1:2);
Resized_Training_Image = augmentedImageDatastore(Input_Layer_Size, Training_Dataset);
Resized_Validation_Image = augmentedImageDatastore(Input_Layer_Size, Validation_Dataset);

%These are being replaces to fit the new data set, as the extracted features now only 
%need to fit one of four specific classifications, instead of the original 1000.
Feature_Learner = net.Layers(142);
Output_Classifier = net.Layers(144);

%Categorize the labels of the dataset and find the number of them.
Number_of_Classes = numel(categories(Training_Dataset.Labels));

%Modify feature learner layer (142)
New_Feature_Learner = fullyConnectedLayer(Number_of_Classes, ...
    'Name', 'Mushroom Types Feature Learner', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

%Modify classify layer (144)
New_Classifier_Layer = classificationLayer('Name', 'Mushroom Types Classifier');

%Storing layer graph in order to modify the network architecture.
Layer_Graph = layerGraph(net);

%Replacing final layers of the network.
New_Layer_Graph = replaceLayer(Layer_Graph, Feature_Learner.Name, New_Feature_Learner);
New_Layer_Graph = replaceLayer(New_Layer_Graph, Output_Classifier.Name, New_Classifier_Layer);

%analyzeNetwork(New_Layer_Graph);



%Calculate validation frequency by flooring the number of files in the
%resized training image. For visuals.
Validation_freq = floor(numel(Resized_Training_Image.Files)/Size_of_Minibatch);

%Training options
options = trainingOptions("sgdm", ...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 6, ...     
    'InitialLearnRate', 3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', Resized_Validation_Image, ... 
    'ValidationFrequency', Validation_freq, ...
    'Verbose', false, ...  
    'Plots','training-progress');

%Command for training network with dataset, modified network architecture,
%and training options.
net = trainNetwork(Resized_Training_Image, New_Layer_Graph, options);


%Classify validation images and calculate the classification accuracy.
[YPred,probs] = classify(net, Resized_Validation_Image);
accuracy = mean(YPred == Validation_Dataset.Labels);

%Plot images showing accuracy of first 9 classified mushrooms.
figure
for i = 1:9
    subplot(3, 3, i);
    I = readimage(Validation_Dataset,i);
    R = imresize(I, [224, 224]);
    imshow(R)
    label = YPred(i);
    if YPred(i) == Validation_Dataset.Labels(i)
        title(string(label) + ", " + num2str(100*max(probs(i,:)),3) + "%", 'Color','#77AC30');
    else
        title(string(label) + ", " + num2str(100*max(probs(i,:)),3) + "%", 'Color','#A2142F');
    end
end
 
%Plot confusion matrix.
figure
plotconfusion(Validation_Dataset.Labels, YPred);



