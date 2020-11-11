def computeCost(X, y, theta):
    """
    Purpose:
    --------
    Computes the cost function for linear regression
    Uses theta as the parameter to fit X and y
    
    Parameters:
    -----------
    Inputs:
    -------
        X:      Dimension 1, feature data
        y:      Dimension 2, label data
        theta:  Initial gradient descent parameters
        
    Outputs:
    --------
        J:      Cost Function Value
    """
    
    # Initialize useful values
    m = len(y)
    J = 0 # holds cost function value
    
    h = X @ theta
    tosquare = h - np.expand_dims(y, axis = 1)
    summation = sum(tosquare ** 2)
    J = summation / (2 * m)
    
    return J