function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.
m = length(y);
%X = [ones(m, 1) X];
%theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE =============================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%
%X = [ones(m, 1), data(:,1)];
%m = length(y)
%X = [ones(m, 1), X];
theta = pinv(X' * X) * X' * y;

% ===================================================================

end
