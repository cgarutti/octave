%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.
function [J, grad] = costFunctionReg(theta, X, y, lambda);
  % Initialize some useful values
  m = length(y); % number of training examples

  % return values
  J = 0;
  grad = zeros(size(theta));

  % Compute the cost J of a particular choice of theta.
  % Compute the partial derivatives grad.
  h = sigmoid(X*theta); % hypothesis in logistic regression
  v = ones(size(theta)); % vectorizing multiplication in the regularization term
  v(1) = 0;
  J = -(1/m)*(y'*log(h)+(1-y)'*log(1-h)) + (lambda/(2*m))*((theta.^2)'*v); % cost function in regularized logistic regression

  v = ones(size(theta));
  v(1) = 0;
  grad = (1/m)*X'*(h-y) + (lambda/m)*(theta.*v); % gradient in logistic regression
end;
