function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% computing the sigmoidal version of h(x) see also sigmoid.m for the sigmoid function
h = sigmoid(X * theta);

% computing the cost function for logistic regression. y is transposed so the items will sum.
J = (1/m) * (-y' * log(h) - ((1 - y') * log(1 - h)));

% computing the gradient of the cost for simple logistic regression.
% mostly identical to the linear regression example, but with different h(x)
grad = (1/m) * (X' * (h - y));

% =============================================================

end
