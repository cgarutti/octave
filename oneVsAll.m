%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta
%corresponds to the classifier for label i
function [all_theta] = oneVsAll(X, y, num_labels, lambda);
  m = size(X, 1); n = size(X, 2); %useful variables
  all_theta = zeros(num_labels, n + 1); %return variables
  X = [ones(m, 1) X]; %add ones to the X data matrix

  % Train num_labels logistic regression classifiers
  % with regularization parameter lambda.
  for c = 1:num_labels;
    initial_theta = zeros(n + 1, 1); %initial theta
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    [theta] = fmincg (@(t)(costFunctionReg(t, X, (y == c), lambda)), initial_theta, options);
    all_theta(c,:)=theta; %assign c-th row
  end;
end;
