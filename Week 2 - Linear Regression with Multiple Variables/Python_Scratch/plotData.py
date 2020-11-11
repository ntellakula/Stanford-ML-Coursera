def plotData(x, y):
    """
    Purpose:
    --------
    plots the data points X and y into a new figure
    gives appropriate axes names
    
    Parameters:
    -----------
    X: Dimension 1, feature data
    y: Dimension 2, label data
    """
    
    fig = plt.figure(figsize = [10, 10])
    plt.scatter(x, y, c = 'red', marker = 'x', s = 25)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')