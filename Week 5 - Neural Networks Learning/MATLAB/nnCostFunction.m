function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
	% NNCOSTFUNCTION Implements the neural network cost function for a two layer
	%	neural network which performs classification
	
	%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
	%   	X, y, lambda) computes the cost and gradient of the neural network.
	%   	The parameters for the neural network are "unrolled" into the vector
	%   	nn_params and need to be converted back into the weight matrices. 
	% 
	%   	The returned parameter grad should be a "unrolled" vector of the
	%   		partial derivatives of the neural network.
	%

	% Reshape nn_params back into the parameters Theta1 and Theta2, the weight
	% matrices for our 2 layer neural network
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
					 hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
					 num_labels, (hidden_layer_size + 1));

	% Setup some useful variables
	m = size(X, 1);
			 
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

	%------------------------------ Part 1 -----------------------------------%
	% Add bias layer of 1s to X
	a1 = [ones(m, 1) X];    % 5000 x 401

	z2 = a1 * Theta1';      % values for the first hidden layer
	a2 = sigmoid(z2);       % 5000 x 25

	% Add bias layer of 1s to a2
	a2 = [ones(size(a2, 1), 1) a2];     % 5000 x 26

	z3 = a2 * Theta2';      % values for the output layer
	a3 = sigmoid(z3);       % 5000 x 10

	% Convert y into a vector of 0s and 1s
	yvec = [1:num_labels] == y;         % 5000 x 10, =1 for output digit

	% Cost function, no regularization, J is a scalar
	J = (1 / m) * sum(sum((-yvec .* log(a3)) - (1 - yvec) .* log(1 - a3)));

	% Remove bias layer from the Thetas
	theta1_nobias = Theta1(:, 2:size(Theta1, 2));   % 25 x 400
	theta2_nobias = Theta2(:, 2:size(Theta2, 2));   % 10 x 25

	% Should be a scalar
	reg = (lambda / (2 * m)) * (sum(sum(theta1_nobias .^ 2)) + sum(sum(theta2_nobias .^ 2)));
	J = J + reg;

	%------------------------------ Part 2 -----------------------------------%
	delta3 = a3 - yvec;     % 5000 x 10
	delta2 = (delta3 * Theta2) .* [ones(size(z2, 1), 1) sigmoidGradient(z2)];

	% Remove bias layer from delta2
	delta2_nobias = delta2(:, 2:size(delta2, 2));   % 5000 x 25

	% Unregularized gradients
	Theta1_grad = (1 / m) * (delta2_nobias' * a1);  % 25 x 401
	Theta2_grad = (1 / m) * (delta3' * a2);         % 10 x 26

	%------------------------------ Part 3 -----------------------------------%
	reg_term1 = (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:size(Theta1, 2))];
	reg_term2 = (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:size(Theta2, 2))];

	% Add the regularization terms to previously calculated gradient
	Theta1_grad = Theta1_grad + reg_term1;
	Theta2_grad = Theta2_grad + reg_term2;

	% -------------------------------------------------------------

	% =========================================================================

	% Unroll gradients
	grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
