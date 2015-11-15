function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
%
%%the below commented out code from from the internet, but I figured out how to do it
%a different way 
%X_poly(:,1) = X;

%for i=2:p
%	X_poly(:,i) = X .* X_poly(:,i-1);
%end
 %this is my code. basically I am just adding X to the ith power to a new matrix, X-poly 
for i=1:p
	X_poly(:,i) = X.^i;
end

% =========================================================================

end
