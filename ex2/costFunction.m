function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

# X: m * f
# y: m * 1
# theta: f * 1
# yPredictions: m * 1
yPredictions = sigmoid(X * theta);

# costForYEq1: m * 1, if the y prediction is 1, cost should be 0, if it's 0, then it should be inf
costForYEq1 = -log(yPredictions);

# costForYEq0: m * 1, if the y prediction is 0, then the cost is 0, if it's 1, then it should be inf
costForYEq0 = -log(1 - yPredictions);

J = (y' * costForYEq1 + (1 - y)' * costForYEq0) / m;

thetaAdjustments = X' * (yPredictions - y) / m;

grad = thetaAdjustments;




% =============================================================

end
