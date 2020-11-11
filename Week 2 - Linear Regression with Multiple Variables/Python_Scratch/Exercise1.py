# Necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set Up Environment, import data
os.chdir('C:/Users/ntell/Documents/GitHub/Stanford-ML-Coursera')
data = pd.read_csv('Week 2 - Linear Regression with Multiple Variables/MATLAB/ex1data1.txt', header = None)

# Separate out X and y, store as NumPy arrays
X = data.iloc[:, 0].to_numpy()
y = data.iloc[:, 1].to_numpy()

# Visualize the data
plotData(X, y)


# Implementation of Gradient Descent
m = len(X) # number of examples
X = np.concatenate([np.expand_dims(np.ones(m), axis = 1),
                    np.expand_dims(X, axis = 1)], axis = 1) # add column for intercepts
theta = np.zeros([2, 1]) # Initializing Fitting Parameters
iterations = 1500
alpha = 0.01

computeCost(X, y, theta)
computeCost(X, y, np.array([-1, 2]).reshape(2, 1))


# Run gradient descent
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

# Plot the linear fit
plt.scatter(X[:, 1], y, c = 'red', marker = 'x', label = 'Training Data')
plt.plot([np.min(X[:, 1]), np.max(X[:, 1])],
         [np.min(X @ theta), np.max(X @ theta)],
         c = 'blue', label = 'Linear Regression')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')

# Predictions
predict1 = ((np.array([1, 3.5]) @ theta) * 10000)[0]
print('For a population of 35k, we predict a profit of: %8.2f' % predict1)

predict2 = ((np.array([1, 7]) @ theta) * 10000)[0]
print('For a population of 35k, we predict a profit of: %8.2f' % predict2)


# Visualize J(theta)