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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%%%%%%%%% what follows is my code   %%%%%%%%%%%%%

X = [ones(m, 1) X];       % add ones to the first column of the data matrix as usual

Ym = [1:num_labels] == y; %create a matrix based on y values so that each y class (num_labels) is a vector of a 1 and 0's

% computing the cost follows:

z2 = X * Theta1';                 % compute z2 - matrix of features (X) times the transpose of parameters/weights (Theta1)
a2 = sigmoid(z2);                 % compute the activation of the hidden layer by taking the sigmoids of values calculated above
a2 = [ones(size(a2,1),1), a2];    % add a column of ones to the beginning of a2 matrix for the bias unit
h = sigmoid(a2 * Theta2');        % calculate the output layer activation - these are also the "calculated y values"

% compute regularization
% the matrices of parameters (Theta1 and Theta2) are individually squared element-wise, skipping the first column
% the columns are summed into a row vector, which is then summed into a scalar, I think it happens in this order
% but it is summed in one direction, then the other to produce the scalar
reg = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% compute the cost via logistic regression cost function
% here the matrix of actual y values (Ym) as vectors is multiplied element-wise by the "calculated y values" (h) 
J = (1/m) * sum(sum((-Ym) .* log(h) - (1 - Ym) .* log(1 - h)));
J += reg;                          % adding the regularization to the cost                             


% computing the back propagation
d3 = h - Ym;
%d2 = (d3 * Theta2 .* sigmoidGradient([ones(size(z2, 1), 1) z2]))(:,2:end);
d2 = (d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);

D1 = d2' * X;
D2 = d3' * a2;

Theta1_grad = D1/m + (lambda/m) * [zeros(size(Theta1,1),1) Theta1(:, 2:end)];
Theta2_grad = D2/m + (lambda/m) * [zeros(size(Theta2,1),1) Theta2(:, 2:end)];










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end