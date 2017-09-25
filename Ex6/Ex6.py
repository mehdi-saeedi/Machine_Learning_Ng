# Ex6 of Andrew Ng's course in Machine Learning

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import svm
from matplotlib.colors import ListedColormap


# plot data based on y value
def plotData(X, y):
    pos = X[np.where(y == 1)[0]]
    neg = X[np.where(y == 0)[0]]

    plt.scatter(pos[:, 0], pos[:, 1], marker="+")
    plt.scatter(neg[:, 0], neg[:, 1], marker="o")


# boundary line visualization
def visualizeBoundaryLinear(X, y, model):
    w = model.coef_.reshape(1, -1)
    b = model.intercept_.reshape(1, -1)
    xp = (np.linspace(min(X[:, 1] - 2), max(X[:, 1] + 2), 100)).reshape(1, -1)
    yp = - (w[0][0] * xp + b) / w[0][1]
    plt.plot(xp.flatten(), yp.flatten())


# here we use contour to visualize boundary
def visualizeBoundary(X, y, model, resolution):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())


# load data and prepare X and y
ex6 = loadmat('ex6data1.mat')
X = ex6["X"]
y = ex6["y"]

# plot training data
plotData(X, y)

# build SVM model: linear
model_linear = svm.SVC(kernel='linear', C=1e1)
model_linear.fit(X, y.flatten())
visualizeBoundaryLinear(X, y, model_linear)

# section 2
plt.figure()

# load data and prepare training data
ex6 = loadmat('ex6data2.mat')
X = ex6["X"]
y = ex6["y"]

# use kernel for svm
plotData(X, y)
model_rpf = svm.SVC(kernel='rbf', C=100, gamma=10, random_state=0)
model_rpf.fit(X, y.flatten())
visualizeBoundary(X, y, model_rpf, 0.02)

# section 3
plt.figure()

# load data and prepare training data
ex6 = loadmat('ex6data3.mat')
X = ex6["X"]
y = ex6["y"]

plotData(X, y)

# use kernel for svm
model_rpf = svm.SVC(kernel='rbf', C=1, gamma=10, random_state=0)
model_rpf.fit(X, y.flatten())
visualizeBoundary(X, y, model_rpf, 0.02)
plt.show()
print('done')
