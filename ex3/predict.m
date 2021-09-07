function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

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

theta1Size = size(Theta1)
theta2Size = size(Theta2)

% Add ones to the X data matrix (this is to add theta0, X size changes to 401)
% Calculate first layer
X = [ones(m, 1) X];
z2 = X*Theta1';
a2 = sigmoid(z2);

% Calculate second layer
a2 = [ones(m, 1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

%size(a3)
predictionsForTheFirstDataRow = a3(1,:) % predictions for each class for the first 20*20 image

% we nee to pick only the index of the largest prediction
[max_values indices] = max(a3, [], 2);
p = indices;







% =========================================================================


end
