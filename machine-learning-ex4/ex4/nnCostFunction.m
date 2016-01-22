function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Setup some useful variables
m = size(X, 1);
X = [ones(m,1) X];
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
vector_y = zeros(m ,num_labels);
for c= 1:num_labels
   vector_y(:,c)=(y==c);
end
        
% a_2_1 = sigmoid(X*Theta1');
% a_2 =[ones(m,1) a_2_1]; %5000x26
% a_3 = sigmoid(a_2*Theta2');
% theta_1_pow = Theta1(:,2:end).^2;
% theta_2_pow = Theta2(:,2:end).^2;
% term_1 = vector_y'*log(a_3);
% term_2=(ones(m,num_labels)-vector_y)'*log(ones(size(a_3))-(a_3));
% J = (-1/m)*(sum(term_1(:))+sum(term_2(:)))+(0.5*lambda/m)*(sum(theta_1_pow(:))+sum(theta_2_pow(:)));
% 
% sigma_3 =a_3 -vector_y; %5000x10
% sigma_2=(sigma_3*Theta2).*(a_2.*(ones(size(a_2))-a_2)); %sigma_3&theta2 is 5000*26  
% % sigma_2 is 5000x26
% sigma_2=sigma_2(:,2:end);
% delta_2 = sigma_3'*a_2; %10x26
% delta_1 =  sigma_2'*X; %401x26
% 
% 
% 
% Theta1_grad=delta_1;
% Theta2_grad=delta_2;
    


a1=X;
z2=(a1*Theta1');
a2=sigmoid(z2);    
a2 = [ones(m,1) a2];      %5000x 26
z3=a2*Theta2';
a3=sigmoid(z3);           % a3 is the Htheta(x) size is 5000x10
total_without_theta = 0;


 theta_1_pow = Theta1(:,2:end).^2;
 theta_2_pow = Theta2(:,2:end).^2;

for i = 1:m
    temp_y = vector_y(i,:);
    temp_output = a3(i,:);
    term_1 = temp_y*log(temp_output)';
    term_2= (ones(size(temp_y))-temp_y)*log(ones(size(temp_output))-temp_output)';
    total_without_theta=total_without_theta+(term_1+term_2);
end
total_without_theta = (-1/m)*total_without_theta;
J = total_without_theta+(0.5*lambda/m)*(sum(theta_1_pow(:))+sum(theta_2_pow(:)));

delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));

for i = 1:m
   temp_y = vector_y(i,:);
   l_3 = a3(i,:)-temp_y(1,:);    % 1x10
   l_2 = (l_3*Theta2).*((a2(i,:)).*(ones(size(a2(i,:)))-a2(i,:))); %1x26
   delta_2 = delta_2 + l_3'*a2(i,:); %10x 26
   delta_1 = delta_1 + l_2(2:end)'*a1(i,:); %25x401
end
temp_Theta1 = Theta1;
temp_Theta2 = Theta2;
temp_Theta1(:,1)=zeros(size(Theta1,1),1);
temp_Theta2(:,1)=zeros(size(Theta2,1),1);


Theta1_grad = (1/m)*delta_1 + (lambda/m).*temp_Theta1;
Theta2_grad = (1/m)*delta_2 + (lambda/m).*temp_Theta2;
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
