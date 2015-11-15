function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% computing the sigmoidal version of h(x) see also sigmoid.m for the sigmoid function
h = sigmoid(X * theta);

% setting the first parameter (theta(1) - theta0 from notes) to 0 so it will
% not be included in the regularization
% note that the h(x) is calculated above with the fitted thetas
theta(1) = 0;

% computing the regularized cost function for logistic regression. y is transposed so the items will sum.
% the theta^2 is set up as transpose, but first parameter (theta0 from notes) is now 0 for the regularization
J = ((1/m) * (-y' * log(h) - ((1 - y') * log(1 - h)))) + ((lambda/(2 * m)) * (theta' * theta ));

% computing the gradient of the cost for regularized logistic regression.
% mostly identical to the linear regression example, but with different h(x), and note that again
% the first parameter is not regularized.

grad = (1/m) * (X' * (h - y)) + (lambda/m) * theta;

% =============================================================

end