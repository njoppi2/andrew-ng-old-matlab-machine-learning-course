function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

# REMEMBER: this is the cost function, not its derivative, the weights adjustments

# X: 97 x 2
# theta: 2 x 1
yPrediction = X * theta;

# prediction: 97 x 1
# y: 97 x 1
J = sum((yPrediction - y).^2) / (2 * m);
# the cost is the sum of all squared differences between the actual y and the predicted y, for each training example..
# divided by the number of training examples * 2




% =========================================================================

end
