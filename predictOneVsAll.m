%PREDICT Predict the label for a trained one-vs-all classifier. The labels
%are in the range 1..K, where K = size(all_theta, 1).
function p = predictOneVsAll(all_theta, X);
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples)
  m = size(X, 1);
  num_labels = size(all_theta, 1);

  p = zeros(size(X, 1), 1); %return values
  X = [ones(m, 1) X]; %add ones to the X data matrix

  % multiply inputs with parameters to find the predicted values
  predictions = X*all_theta';

  % assign the max of the predicted values as the prediction
  [val,p]=max(predictions, [], 2);
end;
