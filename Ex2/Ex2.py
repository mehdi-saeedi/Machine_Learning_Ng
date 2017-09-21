# whether a student gets admitted into a university.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def costFunction(theta, X, Y):
    Z = np.dot(X, theta)
    h = sigmoid(Z)
    J = 1 / m * np.sum(-Y * np.log(h) - (1 - Y) * np.log(1 - h))
    grad = 1 / m * np.dot(X.T, h - Y)
    return J, grad.flatten()


def sigmoid(z):
    z_exp = np.exp(-z)
    result = 1 / (1 + z_exp)
    return result


# get inputs
input_file = open('ex2data1.txt', 'r')
ex2data1 = pd.read_csv(input_file).values
Exam1 = ex2data1[:, 0]
Exam2 = ex2data1[:, 1]
Decision = ex2data1[:, 2]

m = len(Exam1)

Admitted_Exam1 = Exam1[np.where(Decision == 1)]
Admitted_Exam2 = Exam2[np.where(Decision == 1)]
Not_Admitted_Exam1 = Exam1[np.where(Decision == 0)]
Not_Admitted_Exam2 = Exam2[np.where(Decision == 0)]

plt.xlabel('Exam1 Score')
plt.ylabel('Exam2 Score')
Admitted_plt = plt.scatter(Admitted_Exam1, Admitted_Exam2, marker="o", label="Admitted")
Not_Admitted_plt = plt.scatter(Not_Admitted_Exam1, Not_Admitted_Exam2, marker="+", label="Not admitted")
plt.legend(handles=[Admitted_plt, Not_Admitted_plt], loc=1)

# Add one column of 1
X = np.hstack((np.ones((m, 1), dtype=Exam1.dtype), Exam1.reshape(m, 1), Exam2.reshape(m, 1)))
Y = Decision
lambda_ = 0
initial_theta = np.array([0] * 3)

[cost, grad] = costFunction(initial_theta, X, Y)

myargs = (X, Y)
solution = minimize(costFunction, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter': 400},
                    method="Newton-CG", jac=True)
theta = solution['x']

plot_x = np.array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])
plot_y = np.array((-1 / theta[2]) * (theta[1] * plot_x + theta[0])).reshape(2, )

plt.plot(plot_x, plot_y)
print("theta=", theta)

# predict
test = np.array([1, 45, 85])
print("Prob=", sigmoid(np.dot(test, theta)))

# accuracy
p = np.mean((((sigmoid(np.dot(X, theta)) >= 0.5).astype(int)) == Y).astype(int)) * 100
print("Training accuracy=", p)
plt.show(block=True)
