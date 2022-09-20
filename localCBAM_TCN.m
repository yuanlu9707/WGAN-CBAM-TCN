numObservations = numel(XTrain)
numClasses = numel(classes)

numFilters = 512;
filterSize = 5;
dropoutFactor = 0.005;
numBlocks = 4;

Name_input="input"+i;
layer = sequenceInputLayer(numFeatures,'Normalization',"rescale-symmetric",'Name',Name_input);
lgraph = layerGraph(layer);

outputName = layer.Name;

for i = 1:numBlocks
    dilationFactor = 2^(i-1);
    name_cov1="conv1_"+i;
    name_add="add_"+i;
    layers = [
        convolution1dLayer(filterSize,numFilters,'DilationFactor',dilationFactor,'Padding',"causal",'Name',name_cov1)
        layerNormalizationLayer
        spatialDropoutLayer(dropoutFactor)
        convolution1dLayer(filterSize,numFilters,'DilationFactor','dilationFactor','Padding',"causal")
        layerNormalizationLayer
        reluLayer
        spatialDropoutLayer(dropoutFactor,'Name',"cbam_input1")];
    lgraph = addLayers(lgraph,layers);

    layers = [
        globalAveragePooling2dLayer('Name',"avg1_1")
        fullyConnectedLayer(numFilters,'Name',"fc1_1")
        fullyConnectedLayer(numFilters,'Name',"fc1_2")];
    lgraph = addLayers(lgraph,layers);
    
    layers = [
        globalMaxPooling2dLayer('Name',"max1_1")
        fullyConnectedLayer(numFilters,'Name',"fc2_1")
        fullyConnectedLayer(numFilters,'Name',"fc2_2")
        additionLayer(2,'Name','adds')
        sigmoidLayer("Name","sigmoid_1")
        multiplicationLayer(2,"Name","multiplication_1");];
    lgraph = addLayers(lgraph,layers);
    
        layers = [
        convolution2dLayer(1,1,"Name","2dconv_1","Padding","same")
        maxPooling2dLayer(1)
        concatenationLayer(2,2,"Name","concatenation")
        convolution2dLayer(1,1,"Name","2dconv_1","Padding","same")
        sigmoidLayer("Name","sigmoid_2")
        multiplicationLayer(2,"Name","multiplication_2")
        additionLayer(2,'Name',name_add)];
    lgraph = addLayers(lgraph,layers);
    
        layers = [
        averagePooling2dLayer(1,"Name","avg2_2")];
    lgraph = addLayers(lgraph,layers);

    
    % Add and connect layers.
%     lgraph = addLayers(lgraph,layers);
	lgraph = connectLayers(lgraph,"cbam_input1","avg1_1");
    lgraph = connectLayers(lgraph,"cbam_input1","max1_1");
    lgraph = connectLayers(lgraph,"fc1_2","adds/in2");
    lgraph = connectLayers(lgraph,"cbam_input1","multiplication_1/in2");
    lgraph = connectLayers(lgraph,"multiplication_1","2dconv_1");
    lgraph = connectLayers(lgraph,"2dconv_1","avg2_2");
    lgraph = connectLayers(lgraph,"avg2_2","concatenation");
    lgraph = connectLayers(lgraph,"multiplication_1","multiplication_2/in2");
    
    lgraph = connectLayers(lgraph,outputName,"conv1_"+i);

    % Skip connection.
    if i == 1
        % Include convolution in first skip connection.
        layer = convolution1dLayer(1,numFilters,'Name',"convSkip");

        lgraph = addLayers(lgraph,layer);
        lgraph = connectLayers(lgraph,outputName,"convSkip");
        lgraph = connectLayers(lgraph,"convSkip","add_" + i + "/in2");
    else
        lgraph = connectLayers(lgraph,outputName,"add_" + i + "/in2");
    end
    
    % Update layer output name.
    outputName = "add_" + i;
end

layers = [
    fullyConnectedLayer(numClasses,'Name',"fc")
    softmaxLayer
    classificationLayer];
lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,outputName,"fc");


options = trainingOptions("adam", ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',4, ...
    'ValidationData',{XTest_va,YTest_va}, ...
    'MiniBatchSize',2048, ...
    'InitialLearnRate',0.001, ...
    'LearnRateDropPeriod',20, ... 
    'Verbose',0);

net = trainNetwork(XTrain,TTrain,lgraph,options);

