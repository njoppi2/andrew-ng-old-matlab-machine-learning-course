function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%cost
predictedY = X * theta;
error = predictedY - y;
meanSquaredError = (error).^2;
costRegularization = lambda * [0; theta(2:end)].^2; # watch out for the [0; theta(2:end)]
cost = (sum(meanSquaredError) + sum(costRegularization)) / (2 * m);
J = cost;

%gradient
thetaWithoutBiasWeight = [0; theta(2:end)]; # watch out for the [0; theta(2:end)]
gradRegularization = lambda * thetaWithoutBiasWeight;
grad = (X' * error + gradRegularization) / m;


% =========================================================================

grad = grad(:);

end
