function layer = multiplicationLayer(numInputs, optArgs)
% multiplicationLayer   Element-wise Multiplication layer
%
%   layer = multiplicationLayer(numInputs) Creates a multiplication layer that 
%   takes multiple inputs and performs element-wise multiplication. The number 
%   of inputs to the layer is specified by numInputs.
%
%   layer = multiplicationLayer(numInputs, 'PARAM1', VAL1) specifies additional 
%   name value pair arguments that sets the properties (?) of the layer.
%
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%   A multiplication layer has the following inputs:
%       'in1','in2',...,'inN'     - Inputs to be multiplied together. 
%   
%   Note: The size of the inputs to the multiplication layer must be either same 
%   across all dimensions or same across atleast one dimension with other dimensions 
%   as singleton dimensions.
%   
%
%   Example:
%       % Create a multiplication layer with two inputs that element-wise multiplies 
%       % the output from two ReLU layers.
%
%       mul_1 = multiplicationLayer(2,'Name','mul_1');
%       relu_1 = reluLayer('Name','relu_1');
%       relu_2 = reluLayer('Name','relu_2');
%
%       lgraph = layerGraph();
%       lgraph = addLayers(lgraph, relu_1);
%       lgraph = addLayers(lgraph, relu_2);
%       lgraph = addLayers(lgraph, mul_1);
%
%       lgraph = connectLayers(lgraph, 'relu_1', 'mul_1/in1');
%       lgraph = connectLayers(lgraph, 'relu_2', 'mul_1/in2');
%
%       plot(lgraph);
%
%   See also nnet.cnn.layer.MultiplicationLayer, concatenationLayer, additionLayer.
%
%   <a href="matlab:helpview('deeplearning','list_of_layers')">List of Deep Learning Layers</a>

%   Copyright 2020 The MathWorks, Inc.

arguments
    numInputs {iAssertValidNumInputs}
    optArgs.Name {iAssertValidLayerName} = '' 
end

% Convert arguments to canonical form.
inpArgs.Name = char(optArgs.Name);  % make sure strings get converted to char vectors
inpArgs.NumInputs = numInputs;

% Construct an user visible multiplication layer.
layer = nnet.cnn.layer.MultiplicationLayer(inpArgs.Name, inpArgs.NumInputs);

end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end

function iAssertValidNumInputs(value)
validateattributes(value, {'numeric'}, ...
    {'real','finite','positive', 'integer', 'nonempty', 'scalar', '>', 1});
end