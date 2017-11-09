# Q7 Assignment 3

# Import Files
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.decomposition import PCA

# Load Data Set
iris = datasets.load_iris()
# Definte Input and Target Vectors
X = iris.data
y = iris.target

# Using PCA to compute the two best features
X_temp = X
numFeat = 2
pca = PCA(n_components = numFeat)
X_r = pca.fit(X).transform(X)
X = X_r

X = X[y != 0, :numFeat]
y = y[y != 0]

# Randomizing The Data Used For Training And Testing
n_sample = len(X)
np.random.seed(0)
order = np.random.permutation(n_sample)

X = X[order]
y = y[order].astype(np.float)

# Extracting Training And Testing Samples
X_train = X[:int(0.9*n_sample)]
y_train = y[:int(0.9*n_sample)]
X_test = X[int(0.9*n_sample):]
y_test = y[int(0.9*n_sample):]

for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                            edgecolor='k', s=20)

    # Circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
            zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                            linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
    print(kernel)
    plt.title(kernel)
    plt.xlabel("1st PCA Feature")
    plt.ylabel("2nd PCA Feature")

plt.show()
print("Hello World!")
