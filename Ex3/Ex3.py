# Contains 5000 training examples of handwritten digits
# Ex3 of Andrew Ng's course in Machine Learning

import numpy as np
from scipy.io import loadmat
from scipy.special import expit
from scipy.optimize import minimize


# sigmoid function
def sigmoid(z):
    g = np.zeros(z.shape)
    g = expit(z)
    return g


def lrCostFunction(theta, X, y, lambda_):
    m = len(y)  # number of training examples
    J = 0
    grad = np.zeros(theta.shape)

    one = y * np.log(sigmoid(np.dot(X, theta)))
    two = (1 - y) * (np.log(1 - sigmoid(np.dot(X, theta))))
    reg = (float(lambda_) / (2 * m)) * (np.dot(theta[1:], theta[1:].T))
    J = -(1. / m) * (one + two).sum() + reg

    grad = (1. / m) * np.dot(sigmoid(np.dot(X, theta)) - y, X).T + (float(lambda_) / m) * theta

    # the case of j = 0 (recall that grad is a n+1 vector)
    grad_no_regularization = (1. / m) * np.dot(sigmoid(np.dot(X, theta)) - y, X)

    # and then assign only the first element of grad_no_regularization to grad
    grad[0] = grad_no_regularization[0]

    return J, grad.flatten()


# get inputs
ex3 = loadmat('ex3data1.mat')
num_labels = 10
X = ex3['X']
y = ex3['y'].flatten()

lambda_ = 0.1
m, n = X.shape

all_theta = np.zeros((num_labels + 1, n + 1))
X = np.column_stack((np.ones((m, 1)), X))

# optimize for each label
for c in range(0, num_labels):
    print("c=", c)
    initial_theta = np.zeros(n + 1)

    myargs = (X, (y % 10 == c).astype(int), lambda_)

    theta = minimize(lrCostFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter': 13},
                     method="Newton-CG", jac=True)

    all_theta[c, :] = theta["x"].T

prob = np.dot(X, all_theta.T)
predict = np.argmax(sigmoid(np.dot(X, all_theta.T)), axis=1)

print("accuracy =", np.mean((predict == y % 10).astype(int)))
