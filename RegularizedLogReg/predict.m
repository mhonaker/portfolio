function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% calculate hidden layer (layer 2) activations
 a2 = sigmoid(X*Theta1');
 
 % add ones to the matrix for the bias unit
 a2 = [ones(size(a2,1),1) a2];
 
 % calculate the output layer (layer 3)
 h = sigmoid(a2*Theta2');


%determine the maximum value of each row (val), and what column (the classifier) it is in (ind)
[val,ind]=max(h,[],2);

%set p equal to the column number (the classifier)
p=ind;

% =========================================================================


end
