# Contains 5000 training examples of handwritten digits
# Ex5 of Andrew Ng's course in Machine Learning


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize


# Cost function for linear regression
def linearRegCostFunction(Theta, X, y, lambda_, returnGrad):
    m = X.shape[0]
    Theta = Theta.reshape(1, -1)
    htheta = np.dot(X, Theta.T)
    diff = htheta - y
    J = 1 / (2 * m) * np.power(np.linalg.norm(diff), 2)
    regulazation_J = lambda_ / (2 * m) * (np.dot(Theta, Theta.T) - Theta[0][0] * Theta[0][0])
    J = J + regulazation_J

    grad = 1 / m * np.dot(diff.T, X) + lambda_ / m * Theta
    grad[0] = grad[0] - lambda_ / m * Theta[0][0]
    if (returnGrad == True):
        return J, grad.flatten()
    else:
        return J


# optimize cost function
def trainLinearReg(X, y, lambda_):
    Theta = np.zeros((X.shape[1], 1))
    myargs = (X, y, lambda_, True)
    solution = minimize(linearRegCostFunction, x0=Theta.flatten(), args=myargs, options={'disp': True, 'maxiter': 200},
                        method="L-BFGS-B", jac=True)
    theta = solution["x"].T
    return theta


def learningCurve(X, y, Xval, yval, lambda_):
    m = X.shape[0]
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    for i in range(0, m):
        print(i)
        X_sample = X[0:i + 1, :]
        y_sample = y[0:i + 1]
        theta = trainLinearReg(X_sample, y_sample, lambda_).reshape(1, -1)
        error_train[i], = linearRegCostFunction(theta, X_sample, y_sample, 0, False)
        error_val[i], = linearRegCostFunction(theta, Xval, yval, 0, False)
    return error_train, error_val


# extract polynomial features
def polyFeatures(X, p):
    X_poly = np.zeros((X.shape[0], p))
    for i in range(0, p):
        X_poly[:, i] = np.power(X, (i + 1))
    return X_poly


# normalize features based on mean and std
def featureNormalize(X):
    mu = np.mean(X, 0).reshape(1, -1)
    X_norm = X - mu

    sigma = np.std(X_norm, 0, ddof=1).reshape(1, -1)
    X_norm = X_norm / sigma

    return (X_norm, mu, sigma)


# plot utility
def plotFit(X, p, mu, sigma):
    x = np.arange(min(X[:, 1]) - 15, max(X[:, 1]) + 25, 0.05)
    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma
    X_poly = np.column_stack((np.ones((X_poly.shape[0], 1)), X_poly))
    return x, X_poly


# get data
ex5 = loadmat('ex5data1.mat')
X = ex5['X']
y = ex5['y']
Xval = ex5['Xval']
yval = ex5['yval']
Xtest = ex5['Xtest']
ytest = ex5['ytest']

# set parameters
m = X.shape[0]
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
Xval = Xval.reshape(-1, 1)
yval = yval.reshape(-1, 1)
Xtest = Xtest.reshape(-1, 1)
ytest = ytest.reshape(-1, 1)

# plot training data
plt.scatter(X, y)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

lambda_ = 0
Theta = np.ones(2).reshape(1, 2)

# add bias
X = np.column_stack((np.ones((m, 1)), X))
Xval = np.column_stack((np.ones((Xval.shape[0], 1)), Xval))

[error_train, error_val] = learningCurve(X, y, Xval, yval, lambda_)

plt.figure()
plt_train, = plt.plot(range(0, m), error_train, label="Train Error")
plt_val, = plt.plot(range(0, m), error_val, label="Validation Error")
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend(handles=[plt_train, plt_val])

p = 8
X_poly = polyFeatures(X[:, 1], p)
[X_poly, mu, sigma] = featureNormalize(X_poly)
X_poly = np.column_stack((np.ones((m, 1)), X_poly))

X_poly_test = polyFeatures(Xtest[:, 0], p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.column_stack((np.ones((X_poly_test.shape[0], 1)), X_poly_test))

X_poly_val = polyFeatures(Xval[:, 1], p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.column_stack((np.ones((X_poly_val.shape[0], 1)), X_poly_val))

lambda_ = 3
theta = trainLinearReg(X_poly, y, lambda_).reshape(1, -1)

plt.figure()
plt.scatter(X[:, 1], y)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

x, x_poly = plotFit(X, p, mu, sigma)
plt.plot(x, np.dot(x_poly, theta.T), linestyle='--', label="Train Error")
plt.title('Polynomial Regression Fit lambda=1')

[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
plt.figure()
plt_train, = plt.plot(range(0, error_train.shape[0]), error_train, label="Train Error")
plt_val, = plt.plot(range(0, error_val.shape[0]), error_val, label="Validation Error")
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend(handles=[plt_train, plt_val])

plt.show()
print("done")
