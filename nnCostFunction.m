function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

X=[ones(m,1) X]; % add bias (=1st columns of ones)

A2 = sigmoid(X*(Theta1)'); % get A2 params

A2=[ones(m,1) A2]; % add bias (=1st columns of ones)

h = sigmoid(A2*(Theta2)'); % hypothesis in a neural network with 3 layers (input/hidden/output)

%loop over all the training samples
for i=1:m,
	y_i = (1:num_labels)==y(i); %create vector with all '0' but one '1' in correspondence of the real class
	J = J + y_i*log(h(i,:))'+(1-y_i)*log(1-h(i,:))'; %update the cost function
end;

J = -(1/m)*J; %update the cost function

%now add regularization
Theta1_2=Theta1(:,2:end); %ignore first column (bias)
Theta2_2=Theta2(:,2:end); %ignore first column (bias)
J = J + (lambda/(2*m))*(sum(sum(Theta1_2.^2)) + sum(sum(Theta2_2.^2))); %update the cost function

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

%init the Deltas for accumulation
D_1 = zeros(size(Theta1));
D_2 = zeros(size(Theta2));

%loop over all the training samples
for i=1:m,
	%STEP 1 - feedforward
	a_1 = X(i,:)'; %bias already added to X at the beginning
	z_2 = Theta1*a_1;
	a_2 = sigmoid(z_2); %apply sigmoid
	a_2=[1;a_2]; %add bias
	z_3 = Theta2*a_2;
	a_3 = sigmoid(z_3);
	
	%STEP 2 - calculate d_3 (OUT layer)
	y_i = (1:num_labels)==y(i); %create vector with all '0' but one '1' in correspondence of the real class
	d_3 = a_3 - y_i';
	
	%STEP 3 - calculate d_2 (hidden layer)
	d_2 = Theta2_2'*d_3.*sigmoidGradient(z_2); %ignore first column of Theta2 (bias)
	
	%STEP 4 - accumulate the gradient
	D_2 = D_2 + d_3*a_2';
	D_1 = D_1 + d_2*a_1';
end;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%STEP 5 - Unregularized gradient for NN cost function
Theta1_grad = (1/m)*D_1;
Theta2_grad = (1/m)*D_2;

%STEP 6 - Regularize gradient
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1_2; %skip bias
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2_2; %skip bias

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
