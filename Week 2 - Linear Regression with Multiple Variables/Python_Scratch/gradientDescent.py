def gradientDescent (X, y, theta, alpha, num_iters):
    """
    Purpose:
    --------
    Performs gradient descent to correctly learn theta
    Updates theta by taking num_iters iterations gradient steps with 
        learning rate alpha
    
    Parameters:
    -----------
    Inputs:
    -------
        X:      Dimension 1, feature data
        y:      Dimension 2, label data
        theta:  Initial gradient descent parameters
        alpha:  Constant; learning rate
        num_iters: Constant; number of iterations
        
    Output:
    -------
        theta:      Learned, fit linear regression parameters
        J_history:  Array of all cost Function values
    """
    
    # Initialize values
    m = len(y)
    J_history = np.zeros(num_iters)
    
    # Loop through preset iterations and store J
    for i in range(num_iters):
        errors = X @ theta - np.expand_dims(y, axis = 1)
        theta = theta - ((alpha / m) * X.T @ errors)
        
        # Save the cost in the empty array
        J_history[i] = computeCost(X, y, theta)
        
    return theta, J_history