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

    % Calculate Cost function
    h = sigmoid(X*theta);
    v = -y.*log(h) - (1-y).*log(1-h);
    J = (1/m)*sum(v) + (lambda/(2*m))*sum(theta(2:end).^2);
    
    % Calculate gradient for each feature     
    temp = ( h - y )' * X(:,1);
    grad(1) = (1/m) .* temp;
    
    data_size = size(X, 2); 
    for featureNr = 2:data_size
      %featureNr
      temp = ( h - y )' * X(:,featureNr);
      grad(featureNr) = (1/m) .* temp + (lambda/m)*theta(featureNr);
    end




% =============================================================

end
