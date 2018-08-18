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

% token
% FMCAayIwRmjYkZPt

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% residuals
% res = (X*theta)-y;
% sqr = sum(res.^2);
% J = (1/2*m)*sqr;
% JL = (theta(2:end).^2 * lambda)/(2*m);
% J = J+JL;

% residuals
% res =(X*theta)-y;
% %sumResSqu = sum(res.^2);
% %J1 = sumResSqu/(2*m);
% J1 = sum(((X*theta)-y).^2)/(2*m);
% J2 = (theta(2:end).^2 * lambda)/(2*m);
% J(1)=0;
% J = [J1+J2];
% J = J(:);
J = sum ((X * theta - y) .^ 2) / (2*m) + lambda / (2 * m) * sum(theta(2:end) .^2);

% gradient descent
grad = (X' * (X * theta - y)) / m + lambda / m * [0; theta(2:end)];
% G1 = (X'*res)/m;
% G2 = (theta*lambda)/m;
% G2(1) = 0;
% grad=[G1+G2];

% =========================================================================

grad = grad(:);

end
