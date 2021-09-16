function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%



% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


C_values = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_values = [0.01 0.03 0.1 0.3 1 3 10 30];
error_distributoin = zeros(length(C_values) .* length(sigma_values), 3);
iteration = 1;

   
for j=1:length(sigma_values),      
  
    sigma = sigma_values(j);    
        
    for i=1:length(C_values),
      
        C = C_values(i);
            
        % Train on the test data
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        
        % Predict with the cross-validation data
        predictions = svmPredict(model, Xval);
        
        % Calculate the prediction error
        error = mean(double(predictions ~= yval));
        
        % Save the result
        error_distribution(iteration, :) = [C, sigma, error];
        iteration = iteration + 1
    end
end

error_distribution;

[x, index] = min(error_distribution(:, 3));
error_distribution(index, :)

%plot3(error_distribution(:, 1), error_distribution(:, 2), error_distribution(:, 3), 'o-r');
% =========================================================================

% You need to return the following variables correctly.
C = error_distribution(index, 1);
sigma = error_distribution(index, 2);

end
