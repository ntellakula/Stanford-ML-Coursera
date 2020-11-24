def featureNormalize(X):
    """
    Purpose:
    --------
    Normalizes the feature in X such that mean is 0 and standard deviation is
        1 for each feature in X. 
        
    Parameters:
    -----------
    Inputs:
    -------
        X: Feature data
        
    Outputs:
    --------
        X_norm:     Normalized version of X
        mu:         Mean of each feature in X
        sigma:      Standard deviation of each feature in X
    """
    
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma