# predict profits for a food truck.
# Ex1 of Andrew Ng's course in Machine Learning

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# Cost calculation function
def computeCost(X, y, theta):
    H = np.dot(X, theta)
    J = 1 / (2 * m) * sum(pow(H - y, 2))
    return J


# get X & y from the input text file
input_file = open('ex1data1.txt', 'r')
ex1data1 = pd.read_csv(input_file).values
X = ex1data1[:, 0].reshape(-1, 1)
y = ex1data1[:, 1]

m = len(X)
Iteration = 1500

# Add one column of 1
X = np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))

theta = np.array([0, 0])
alpha = 0.01

# linear regression
cost_func = np.zeros((Iteration, 1))
for i in range(0, Iteration):
    cost_func[i, 0] = computeCost(X, y, theta)
    theta = theta - alpha / m * np.dot(X.T, np.dot(X, theta) - y)

print(theta)

# plot training data and regressed line
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
trained_plt = plt.scatter(np.array(X[:, 1]), y, label="Training data")
regressed_plt, = plt.plot(X[:, 1], np.dot(X, theta), linestyle='--', label="Linear regression")
plt.legend(handles=[trained_plt, regressed_plt])

plt.show(block=False)

# plot cost function vs iteration to see convergence
plt.figure()
x_cross = np.array([[1, 3.5], [1, 7.0]])
y_cross = np.dot(x_cross, theta)
cost_plt, = plt.plot(cost_func, label="Cost Function")
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend(handles=[cost_plt])

# plot a mesh to see costs for various points
theta0_values = np.arange(-10, 10, 0.2)
theta1_values = np.arange(-1, 4, 0.05)
J_values = np.zeros((100, 100))
for i in range(0, 100):
    for j in range(0, 100):
        t = np.array([theta0_values[i], theta1_values[j]])
        J_values[i, j] = computeCost(X, y, t)

fig = plt.figure()
ax = Axes3D(fig)
theta0_values, theta1_values = np.meshgrid(theta0_values, theta1_values)
surf = ax.plot_surface(theta1_values, theta0_values, J_values, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_0$')

# plot contours for different costs
fig = plt.figure()
CS = plt.contour(theta0_values, theta1_values, J_values, np.logspace(-2, 3, 20),
                 linewidths=np.arange(.5, 4, .5),
                 colors=('r', 'green', 'blue', (1, 1, 0), '#afeeee', '0.5')
                 )
plt.clabel(CS, inline=1, fontsize=10)

plt.show()
