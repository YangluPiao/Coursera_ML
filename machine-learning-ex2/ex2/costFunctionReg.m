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

updates=sigmoid(X*theta);
d=-y'*(log(updates))-(1-y')*(log(1-updates));
J=(1/m)*sum(d)+lambda/(2*m)*sum(theta(2:size(theta),1).^2);%DONOT regularize theta_0.
grad=(1/m)*X'*(updates-y)+lambda/m*[0;theta(2:size(theta,1),1)];




% =============================================================

end
