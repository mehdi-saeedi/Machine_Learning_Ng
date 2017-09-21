# predict the prices of houses
# Ex1 (multiple variables) of Andrew Ng's course in Machine Learning

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Cost calculation function
def computeCost(X, y, theta):
    H = np.dot(X, theta)
    J = 1/(2*m)*sum(pow(H-y, 2))
    return J

def featureNormalize(x):
    sigma = np.std(x, axis=0)
    miu = np.mean(x, axis=0)
    x = (x - miu)/sigma
    return x, sigma, miu

# get inputs from input file
input_file = open('ex1data2.txt', 'r')
ex1data2 = pd.read_csv(input_file).values
X = ex1data2[:, 0:2].reshape(-1,2)
y = ex1data2[:, 2]

m = len(X)
Iteration = 400

# first normalize features
[X, mu, sigma] = featureNormalize(X)

# Add one column of 1
X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))

theta = np.array([0, 0, 0])
alpha = 0.01

cost_func = np.zeros((Iteration,1))
for i in range(0, Iteration):
    theta = theta - alpha / m * np.dot(X.T, np.dot(X, theta) - y)
    cost_func[i, 0] = computeCost(X, y, theta)

print(theta)

plt.figure()
x_cross = np.array([1, 1650, 3])
y_cross = np.dot(x_cross, theta)

cost_plt, = plt.plot(cost_func, label="Cost Function")
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend(handles=[cost_plt])


plt.show()
print(y_cross)
