function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%calculate h of x, basically what y is calculated to be with some parameters(theta)
h = X*theta;

%replace the first column of theta with zeros, so that they are not regularized in the cost function
%or gradient
theta(1) = 0;

%compute the regularized linear regression cost. note the use of h for the first term where
%theta would appear, but the theta with zeros in the regularization term (after the +)
J = (1/(2*m)) * sum((h - y).^2) + (lambda/(2*m)) * sum(theta.^2);

%compute the gradient for the linear regression. Basically this is the partial derivative 
%of the cost function with respect to theta
grad = (1/m) * (X' * (h - y)) + (lambda/m) * theta;

% =========================================================================

grad = grad(:);

end
