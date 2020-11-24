def computeCostMulti(X, y, theta):
    """
    Purpose:
    --------
    Computes the cost function for linear regression with multiple variables
    Uses theta as the parameter to fit X and y
    
    Parameters:
    -----------
    Inputs:
    -------
        X:      Feature data
        y:      Label data
        theta:  Initial gradient descent parameters
        
    Outputs:
    --------
        J:      Cost Function Value
    """
    
    m = len(y)
    J = (sum(((X @ theta) - np.expand_dims(y, axis = 1)) ** 2)) / (2 * m)
    
    return J