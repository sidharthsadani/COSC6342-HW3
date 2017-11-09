# Question 8 : Own Kernel

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn import svm
from sklearn.model_selection import cross_val_score
# from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate

def my_kernel(X, Y):
    Xt = np.sum(np.square(X),1)
    X1 = np.zeros([X.shape[0],1], dtype=np.float)
    for i in range(len(Xt)):
        X1[i,0] = Xt[i]
    Yt = np.sum(np.square(Y),1)
    Y1 = np.zeros([Y.shape[0],1], dtype=np.float)
    for i in range(len(Yt)):
        Y1[i,0] = Yt[i]
    # return np.dot(X1, Y1.T)
    return np.dot(X1, Y1.T)

#Building A Toy Dataset
X, y = make_circles(factor = 0.7, noise = 0.05, n_samples = 201)

fig = plt.figure(figsize=(7,7))
# ax = fig.add_subplot(1,1,1)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.scatter(X[:,0], X[:,1], c = y, s = 20, edgecolor='k')
ax.set_title("Original Data")
# ax.set_xticks(())
# ax.set_yticks(())
# plt.show()

# Randomizing The Data For Training & Testing
n_sample = len(X)
np.random.seed(0)
order = np.random.permutation(n_sample)

X = X[order]
y = y[order]

# Extracting Training And Testing Samples
X_train = X[:int(0.9*n_sample)]
y_train = y[:int(0.9*n_sample)]
X_test = X[int(0.9*n_sample):]
y_test = y[int(0.9*n_sample):]

# clf = svm.SVC(kernel=my_kernel)
clf = svm.SVC(kernel=my_kernel)
clf.fit(X_train, y_train)

fig2 = plt.figure(figsize=(7,7))
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.scatter(X[:,0], X[:,1], c=y, zorder=10, cmap=plt.cm.Paired,
        edgecolor='k', s=20)

# Circling Out Test Data
ax2.scatter(X_test[:,0], X_test[:,1], s=80, facecolors='none',
        zorder=10, edgecolor='k')

x_min = X[:,0].min()
x_max = X[:,0].max()
y_min = X[:,1].min()
y_max = X[:,1].max()

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)

ax2.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
ax2.contour(XX,YY, Z, colors = ['k', 'k', 'k',],
        linestyles=['--', '-', '--'], levels = [-.5, 0, .5])
ax2.set_title("My Kernel")

scoring = ['precision_macro', 'recall_macro', 'accuracy']
measures = ['test_precision_macro', 'test_recall_macro', 'test_accuracy']

# scores1 = cross_val_score(clf, X, y, scoring=scoring, cv=10)
scores1 = cross_validate(clf, X, y, scoring=scoring, cv=10)
clf2 = svm.SVC(kernel='linear')
#scores2 = cross_val_score(clf2, X, y, cv=10, scoring=scoring)
scores2 = cross_validate(clf2, X, y, cv=10, scoring=scoring)
print("Results[Precision, Recall, Accuracy] for My Kernel vs Linear Kernel")
for measure in measures:
    print(measure, ":")
    print("My Kernel")
    print(scores1[measure])
    print("Linear Kernel")
    print(scores2[measure])
# print(scores1)
# print(scores2)
plt.show()
print("Hello World!")
