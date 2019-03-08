% Note: pre-requisite for this function is the cost function (computeCost).
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);
  % Initialize some useful values
  m = length(y); % num of training examples
  J_history = zeros(num_iters, 1);
  for iter = 1:num_iters;
    theta = theta - alpha*(1/m)*(X'*(X*theta-y));
  J_history(iter) = computeCost(X, y, theta); %save the cost J in every iteration
  end;
end;
