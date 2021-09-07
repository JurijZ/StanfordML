function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

% images of size 20*20 gives 400 points - these are our features (columns in the X dataset)
% this also means that we are going to have 400 theta parameters + theta0; altogether 401 parameters
m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix (this is to add theta0, X size changes to 401)
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% we used to have theta as a vector, 
% but it's ok to make a matrix of theta vectors 
% and to do the multiplication for all of the theta vectors in one go
temp = sigmoid(X*all_theta');
size(temp)
temp(1,:) % predictions for each class for the first 20*20 image

% we nee to pick only the index of the largest prediction
[max_values indices] = max(temp, [], 2);
p = indices;






% =========================================================================


end
