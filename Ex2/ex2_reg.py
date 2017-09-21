# whether microchips from a fabrication plant passes quality assurance (QA).
# Ex2 (with regularization) of Andrew Ng's course in Machine Learning

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


# sigmoid function
def sigmoid(z):
    z_exp = np.exp(-z)
    result = 1 / (1 + z_exp)
    return result

# Polynomial features extraction
def map_features(x1, x2, k):
    degree = 6
    out = np.ones(x1.shape).reshape(-1, 1)
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            T = np.power(x1, i - j) * np.power(x2, j)
            out = np.hstack((T.reshape(-1, 1), out))
    return out


# logistic regression cost function with regularization
def costFunctionReg(theta, X, Y, lambda_):
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))
    J = 1 / m * np.sum(-Y * np.log(h) - (1 - Y) * np.log(1 - h)) + \
        lambda_ / (2 * m) * (np.dot(theta, theta.T) - theta[0] * theta[0])

    G = (lambda_ / m) * theta
    G[0] = 0

    grad = 1 / m * np.dot(X.T, h - Y) + G
    return J, grad.flatten()


# get inputs
input_file = open('ex2data2.txt', 'r')
ex2data2 = pd.read_csv(input_file).values
Exam1 = ex2data2[:, 0]
Exam2 = ex2data2[:, 1]
Decision = ex2data2[:, 2]

# number of training data
m = len(Exam1)
lambda_ = 1

# prepare data for plot
Accepted_Exam1 = Exam1[np.where(Decision == 1)]
Accepted_Exam2 = Exam2[np.where(Decision == 1)]
Rejected_Exam1 = Exam1[np.where(Decision == 0)]
Rejected_Exam2 = Exam2[np.where(Decision == 0)]
plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
Admitted_plt = plt.scatter(Accepted_Exam1, Accepted_Exam2, marker="o", label="Accepted")
Not_Admitted_plt = plt.scatter(Rejected_Exam1, Rejected_Exam2, marker="+", label="Rejected")
plt.legend(handles=[Admitted_plt, Not_Admitted_plt], loc=1)

X = map_features(Exam1, Exam2, len(Exam1))
y = Decision

initial_theta = np.array([0] * X.shape[1])

cost, grad = costFunctionReg(initial_theta, X, y, lambda_)

# minimize the cost function by using Newton-CG method
myargs = (X, y, lambda_)
solution = minimize(costFunctionReg, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter': 400},
                    method="Newton-CG", jac=True)
theta = solution['x']

print("theta=", theta)

# Define the ranges of the grid
u = np.linspace(-1, 1.5, 200)
v = np.linspace(-1, 1.5, 200)

# Initialize space for the values to be plotted
z = np.zeros((len(u), len(v)))

for i in range(0, len(u)):
    for j in range(0, len(v)):
        # Notice the order of j, i here!
        T = map_features(u[i], v[j], 1)
        z[i, j] = np.dot(T, theta)

z = z.T  # important to transpose z before calling contour

# only level 0
plt.contour(u, v, z, [0])

plt.show()
