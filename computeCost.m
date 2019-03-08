%COMPUTECOST Compute cost for linear regression with multiple variables
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
function J = computeCost(X, y, theta);
  %Initialize some useful values
  m = length(y); %num of training examples
  J = 0; %return value

  predictions = X*theta; %compute the cost of a particular choice of theta
  E = (predictions - y); %errors
  J = 1/(2*m)*E'*E; %set J to the cost
end;
