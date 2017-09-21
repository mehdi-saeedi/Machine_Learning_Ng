# Contains 5000 training examples of handwritten digits
# Ex4 of Andrew Ng's course in Machine Learning

import numpy as np
from scipy.io import loadmat
from scipy.special import expit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# sigmoid function
def sigmoid(z):
    g = expit(z)
    return g


# Gradient of the sigmoid function
def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12
    return np.random.random((L_out, 1 + L_in)) * 2 * epsilon_init - epsilon_init


def nnCostFunction(nn_params, input_layer_size, num_labels, hidden_layer_size, y, X, lambda_):
    # extract the parameters
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))
    Theta2 = nn_params[((hidden_layer_size * (input_layer_size + 1))):].reshape(num_labels, (hidden_layer_size + 1))

    m, n = X.shape

    # First layer
    a1 = X

    # Second layer
    z2 = np.dot(X, Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.column_stack((np.ones((m, 1)), a2))  # add bias

    # Third layer
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)
    J = 0
    htheta = a3

    # Feedforward algorithm
    for c in range(0, num_labels):
        y_k = (y == (c + 1)).astype(int)
        hthetak = htheta[:, c]

        result1 = -np.dot(y_k, np.log(hthetak))
        result2 = - np.dot(1 - y_k, np.log(1 - hthetak))
        result = 1 / m * (np.sum(result1 + result2))
        J = J + result

    regularization = lambda_ / (2 * m) * (np.sum(np.power(Theta1[:, 1:], 2)) + np.sum(np.power(Theta2[:, 1:], 2)))
    J = regularization + J

    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # backpropagation algorithm
    # Implement regularization with the cost function and gradients
    for t in range(0, m):
        delta_3 = np.zeros(num_labels)
        for k in range(0, num_labels):
            yk = (y[t] == (k + 1)).astype(int)
            delta_3[k] = (htheta[t, k] - yk)
        delta_3 = delta_3.reshape(1, num_labels)

        delta_2 = np.dot(Theta2.T, delta_3.T) * sigmoidGradient(np.insert(z2[t, :], 0, 1).reshape(1, Theta2.shape[1])).T
        delta_2 = delta_2[1:]

        Theta1_grad = Theta1_grad + delta_2 * a1[t, :].reshape(-1, 1).T
        Theta2_grad = Theta2_grad + delta_3.T * a2[t, :].reshape(1, -1)

    Theta1_grad = Theta1_grad / m
    Theta2_grad = Theta2_grad / m

    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + lambda_ / m * Theta1[:, 1:]
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + lambda_ / m * Theta2[:, 1:]

    grad = np.append(Theta1_grad.flatten(), Theta2_grad.flatten(), axis=0)

    return J, grad


# Predict the label of an input given a trained neural network
def predict(Theta1, Theta2, X):
    z2 = np.dot(X, Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.column_stack((np.ones((m, 1)), a2))  # add bias

    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)

    P = (np.argmax(a3, 1) + 1).reshape(m, 1)
    return P


# get data
ex4 = loadmat('ex4weights.mat')
Theta1 = ex4['Theta1']
Theta2 = ex4['Theta2']
ex4 = loadmat('ex4data1.mat')
X = ex4['X']
y = ex4['y']
y = y.flatten()

# set parameters
input_layer_size = 400
hidden_layer_size = 25
m, n = X.shape
lambda_ = 1
num_labels = Theta2.shape[0]

# plot a few ones
f, axarr = plt.subplots(10, 10)
for i in range(0, 10):
    for j in range(0, 10):
        axarr[i][j].imshow(X[np.random.randint(0, m), :].reshape(20, 20).T, cmap=cm.Greys_r)
        axarr[i][j].axis('off')

# add bias
X = np.column_stack((np.ones((m, 1)), X))

nn_params = np.append(Theta1.flatten(), Theta2.flatten(), axis=0)
J, grad = nnCostFunction(nn_params, input_layer_size, num_labels, hidden_layer_size, y, X, lambda_)

# Initial random weights
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_params = np.append(initial_Theta1.flatten(), initial_Theta2.flatten(), axis=0)

# optimization
myargs = (input_layer_size, num_labels, hidden_layer_size, y, X, lambda_)
solution = minimize(nnCostFunction, x0=initial_nn_params, args=myargs, options={'disp': True, 'maxiter': 20},
                    method="Newton-CG", jac=True)

theta = solution["x"].T

Theta1 = theta[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))
Theta2 = theta[((hidden_layer_size * (input_layer_size + 1))):].reshape(num_labels, (hidden_layer_size + 1))

# Predict the output and check accuracy
P = predict(Theta1, Theta2, X)
accuracy = (P == y.reshape(m, 1)).astype(int)
print("accuracy = ", np.mean(accuracy))

plt.show()
