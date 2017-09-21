# Contains 5000 training examples of handwritten digits
# Ex3 (part 2) of Andrew Ng's course in Machine Learning


import numpy as np
from scipy.io import loadmat
from scipy.special import expit


def sigmoid(z):
    g = np.zeros(z.shape)
    g = expit(z)
    return g


ex3 = loadmat('ex3weights.mat')
Theta1 = ex3['Theta1']
Theta2 = ex3['Theta2']

ex3 = loadmat('ex3data1.mat')
X = ex3['X']
y = ex3['y']

m, n = X.shape

X = np.column_stack((np.ones((m, 1)), X))

z2 = np.dot(X, Theta1.T)
a2 = sigmoid(z2)
a2 = np.column_stack((np.ones((m, 1)), a2))  # add bias

z3 = np.dot(a2, Theta2.T)
a3 = sigmoid(z3)

predict = (np.argmax(a3, 1) + 1).reshape(m, 1)
accuracy = (predict == y).astype(int)
print("accuracy = ", np.mean(accuracy))

print("done")
