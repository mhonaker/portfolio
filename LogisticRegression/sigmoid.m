function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ================================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

g = 1./(1 + exp(-z));

%Note that this works on a scalar, matrix or vector...but is
%done in an element wise manner. This is the way it is supposed to be.

% ========================================================================

end
