function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
C_array = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_array =[0.01 0.03 0.1 0.3 1 3 10 30];
m = size(C_array,2);
n=size(sigma_array,2);
% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
init_model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
init_prediction = svmPredict(init_model, Xval);
init_error = mean(double(init_prediction ~= yval));
for i =1 :m
    for j = 1:n
       temp_C = C_array(i);
       temp_sigma= sigma_array(j);
       model= svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma));
       predictions = svmPredict(model, Xval);
       error = mean(double(predictions ~= yval));
       if error < init_error
          init_error=error;
          C=temp_C;
          sigma=temp_sigma;
       end
    end
end






% =========================================================================

end
