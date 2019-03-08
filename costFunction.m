%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
function [J, grad] = costFunction(theta, X, y);
  % Initialize some useful values
  m = length(y); % number of training examples

  % return values
  J = 0;
  grad = zeros(size(theta));

  % Compute the cost J of a particular choice of theta.
  % Compute the partial derivatives grad.
  h = sigmoid(X*theta); % hypothesis in logistic regression
  J = -(1/m)*(y'*log(h)(1-y)'*log(1-h)); % cost function in logistic regression
  grad = (1/m)*X'*(h-y); % gradient in logistic regression
end;
