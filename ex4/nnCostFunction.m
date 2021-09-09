function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% Caclulate h
% Note - the first row of Theta1 corresponds to the first hidden unit in the second layer.
X = [ones(m, 1) X];
a2 = sigmoid(X*Theta1');
size_of_a2 = size(a2);

a2 = [ones(m, 1) a2];
h = sigmoid(a2*Theta2');
size_of_h = size(h);
%sprintf("%5.4f ", h(1,:))

% Transform y to binary matrix, e.g. 10 -> [0 0 0 0 0 0 0 0 0 1]
Y = [1:num_labels] == y;
%Y(1,:)

% Calculate Cost function
v = -Y.*log(h) - (1-Y).*log(1-h);
% v is a 5000 * 10 matrix, where 10 is the number of outputs, and each value is a delta
%sprintf("%5.4f ", v(1,:))
% so we add all deltas of the output values - sum(v)
% and then calculate an average value over all samples - sum(sum(v))
J = (1/m)*sum(sum(v, 2));

% Regularize the cost function
%size_of_Theta1 = size(Theta1)
%size_of_Theta2 = size(Theta2)
reg = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J = J + reg;

% -------------------------------------------------------------
% Backpropagation implementation in the loop
%x = [1; X(1,:)]

Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));
for t = 1:m
  % Step1
  x = [X(t,:)]; % X is already with added X0=1
  z2 = x * Theta1';
  a2 = sigmoid(z2);
  %size_of_a2 = size(a2)

  a2 = [1, a2];
  z3 = a2 * Theta2';
  h = sigmoid(z3);
  %size_of_h = size(h)
  
  % Step2
  %sprintf("%5.4f ", h(1,:))
  %sprintf("%5.4f ", Y(t,:))  
  d3 = h - Y(t,:);  
  
  %Step3
  t = Theta2' * d3'; % 26x10 * 10x1 = 26x1
  d2 = t(2:end, :)' .* sigmoidGradient(z2);
  %d2_size = size(d2)
  
  %Step4
  Delta_1 = Delta_1 + d2' * x;
  Delta_2 = Delta_2 + d3' * a2;
  
end

Theta1_grad = (1/m)*Delta_1 + (lambda/m)*[zeros(size(Theta1, 1), 1), Theta1(:,2:end)];
Theta2_grad = (1/m)*Delta_2 + (lambda/m)*[zeros(size(Theta2, 1), 1), Theta2(:,2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
