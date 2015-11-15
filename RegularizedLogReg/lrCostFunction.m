function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
%
%  The code below (from me) is the same as the code that I wrote for the regularized 
%  logistic regression cost function, which I had already vectorized.


% computing the sigmoidal version of h(x) see also sigmoid.m for the sigmoid function
h = sigmoid(X * theta);

% setting the first parameter (theta(1) = theta0 from notes) to 0 so it will not be included in the regularization
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

grad = grad(:);

end
