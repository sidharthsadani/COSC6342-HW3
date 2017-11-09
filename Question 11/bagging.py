from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

#using this for the other data set needed -- scikit leanr already included dataset
import sklearn.datasets as data_sets
#using the iris dataset from scikit learn
irisX, irisY = data_sets.load_iris(return_X_y=True)


contents = []
#loading and reading iosphere dataset with comma separated format with newline
with open("iosphere_data.txt") as f:
    for line in f:
        #adding to contents array and spliting from ',' and stripping with newline
        contents.append([s for s in line.strip().split(',')])

X = [[float(p) for p in ex[:-1]] for ex in contents]
Y = [1.0 if sublist[-1] == 'g' else 0.0 for sublist in contents]
print([s[-1] for s in contents])


print(X)
print(Y)

X = np.array(X)
Y = np.array(Y)

bagging = BaggingClassifier(KNeighborsClassifier(),)