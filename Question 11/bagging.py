
#some resources used

#scikit documentation for bagging
#http://scikit-learn.org/stable/modules/ensemble.html#bagging

#youtube tutorial for scikit learn ensemble methods
#https://www.youtube.com/watch?v=NqdyfMbVo1Q&t=5s

#Github tutorial for ensemble methods
#https://github.com/knathanieltucker/bit-of-data-science-and-scikit-learn/blob/master/notebooks/EnsembleMethods.ipynb

#plotting example in scikit learn
#http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

#Team Members
#Chieh Chen ppsid 0837931
#Sidharth Sadani

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

#matplot library from example in scikit learn
import matplotlib.pyplot as plt

import numpy as np

#using this for the other data set needed -- scikit leanr already included dataset
import sklearn.datasets as data_sets
#using the iris dataset from scikit learn
Iris_X, Iris_y = data_sets.load_iris(return_X_y=True)


contents = []
#loading and reading iosphere dataset with comma separated format with newline
with open("iosphere_data.txt") as f:
    for line in f:
        #adding to contents array and spliting from ',' and stripping with newline
        contents.append([s for s in line.strip().split(',')])

X = [[float(p) for p in ex[:-1]] for ex in contents]
Y = [True if sublist[-1] == 'g' else False for sublist in contents]
print([s[-1] for s in contents])


print(X)
print(Y)

X = np.array(X)
Y = np.array(Y)

KIris = KNeighborsClassifier()
RIris = RandomForestClassifier()
DIris = DecisionTreeClassifier()

#set with max_sample at .5 nd max feature at .5
baggingKNeighorIris = BaggingClassifier(
                KIris,
                max_samples=.5,
                max_features=.5)

baggingKRandomForestIris = BaggingClassifier(
                RIris,
                max_samples=.5,
                max_features=.5)

baggingDTreetIris = BaggingClassifier(
                DIris,
                max_samples=.5,
                max_features=.5)


mIosphere = KNeighborsClassifier(n_neighbors=3)

baggingIosphere = BaggingClassifier(
                mIosphere,
                max_samples=.5,
                max_features=2,
                n_jobs=2,
                oob_score=True)

#using cross_valication score
from sklearn.model_selection import cross_val_score
#bagging the iris dataset
baggingIrisbaggingKNeighorIrisScore = cross_val_score(baggingKNeighorIris , Iris_X, Iris_y)
baggingIrisbaggingRandomForestIrisScore = cross_val_score(baggingKRandomForestIris , Iris_X, Iris_y)
baggingDTreetIrisScore = cross_val_score(baggingDTreetIris , Iris_X, Iris_y)

#print bagging score mean


print(baggingIrisbaggingKNeighorIrisScore.mean())
print(baggingIrisbaggingRandomForestIrisScore.mean())
print(baggingDTreetIrisScore.mean())



