%PREDICT Predict whether the label is 0 or 1 using learned logistic
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
function p = predictLogReg(theta, X);
  m = size(X, 1); %number of training examples
  p = zeros(m, 1); %return variable

  % Make predictions using the learned logistic regression parameters.
  h = sigmoid(X*theta); % hypothesis in logistic regression
  pos = find(h>=0.5);
  p(pos) = 1;
end;
