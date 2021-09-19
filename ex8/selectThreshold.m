function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions


% y == 1 - anomaly
% y == 0 - normal

% We consider an anomaly (y = 1) if epsilon is higher then the prediction
predictions = (pval < epsilon);

% if we decide that value should be 1, but in reality it was 0, then it's a false positive
false_positive = sum((predictions == 1) & (yval == 0));

% if we decide that value should be 0, but in reality it was 1, then it's a false negative
false_negative = sum((predictions == 0) & (yval == 1));

% if we decide that value should be 1, and in reality it was 0, then it's a true positive
true_positive = sum((predictions == 1) & (yval == 1));

precision = true_positive / (true_positive + false_positive);
recall = true_positive / (true_positive + false_negative);

F1 = 2 * precision * recall / (precision + recall);


    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
