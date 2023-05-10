classdef MultiplicationLayer < nnet.layer.Layer & nnet.internal.cnn.layer.Traceable ...
        & nnet.internal.cnn.layer.CPUFusableLayer ...
        & nnet.internal.cnn.layer.BackwardOptional ...  
    & nnet.layer.Formattable
% MultiplicationLayer   Multiplication layer
    %
    %   To create a multiplication layer, use multiplicationLayer.
    %
    %   This layer takes multiple inputs with the same or similar channel dimensions and
    %   multiplies them together.
    %
    %   MultiplicationLayer properties:
    %       Name                   - A name for the layer.
    %       Description            - A small description about the layer.
    %       Type                   - Specifies type of the layer i.e.
    %                                element-wise multiplication.
    %       NumInputs              - The number of inputs of the layer.
    %       InputNames             - The names of the inputs of the layer.
    %       NumOutputs             - The number of outputs of the layer.
    %       OutputNames            - The names of the outputs of the layer.
    %
    %   Example:
    %       Create a multiplication layer.
    %
    %       layer = multiplicationLayer(3);
    %
    %       Or, layer = multiplicationLayer(3, 'Name', 'mul');
    %
    %   See also convolution2dLayer, convolution3dLayer
    
    %   Copyright 2020-2021 The MathWorks, Inc.

    methods
        function layer = MultiplicationLayer(name, numInputs)
            layer.Name = name;
            layer.NumInputs = numInputs;
            layer.Description = iGetMessageString( 'nnet_cnn:layer:MultiplicationLayer:oneLineDisplay', numInputs );
            layer.Type = iGetMessageString( 'nnet_cnn:layer:MultiplicationLayer:Type' );
        end
        
        function Z = predict(layer, varargin)
            if isempty(varargin)
                error('MATLAB:narginchk:notEnoughInputs',"Not enough input arguments");
            end
            idx = 1:layer.NumInputs;
            Z = iMultiplyInputs(varargin, idx);
        end
        
        function varargout = backward(layer, varargin)
            % Backward propagate the derivative of the loss function through 
            % the layer.
            varargout = cell(1,layer.NumInputs);
            X = varargin(1: layer.NumInputs);
            k = layer.NumInputs + layer.NumOutputs + 1;
            dLdZ = varargin{ k };
            for i = 1:layer.NumInputs
                idx = [1:i-1, i+1:layer.NumInputs];
                val = iMultiplyInputs(X,idx) .* dLdZ;
                % Finding singleton dimensions of input.
                singletonDim = size(X{i})==1;        
                f = find(singletonDim);
                if f
                   % Sum up all the derivative values along those singleton
                   % singleton dimesions so that size of derivative matches     
                   % to size of the corresponding input.  
                   varargout{i} = sum(val,f);        
                else
                   varargout{i} = val;
                end
            end            
        end
    end
    
    methods(Static = true, Hidden = true)
        function name = matlabCodegenRedirect(~)
            name = 'nnet.internal.cnn.coder.MultiplicationLayer';
        end
    end
    
    methods (Hidden)
        function layerArgs = getFusedArguments(layer)
            % getFusedArguments  Returned the arguments needed to call the
            % layer in a fused network.
            layerArgs = { 'multiplication', layer.NumInputs };
        end

        function tf = isFusable(~, ~, ~)
            % isFusable  Indicates if the layer is fusable in a given network.
            tf = true;
        end
    end

end

function Z = iMultiplyInputs(X,ind)
% Perform element-wise multiplication of the tensors in cell array X that
% correspond to the indices in ind.
Z = X{ind(1)};
for i = ind(2:end)
    Z = Z .* X{i};
end
end

function messageString = iGetMessageString( varargin )
messageString = getString( message( varargin{:} ) );
end