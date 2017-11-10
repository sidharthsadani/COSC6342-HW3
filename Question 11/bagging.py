
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
Y = [1 if sublist[-1] == 'g' else 0 for sublist in contents]

#print below to show the result of the cleaned x and y data from iosphere dataset

#print([s[-1] for s in contents])
#print(X)
#print(Y)


#casting iosphere x and y onto numpy array

IosphereX = np.array(X)
IosphereY = np.array(Y)

KClass = KNeighborsClassifier()
RClass = RandomForestClassifier()
DClass = DecisionTreeClassifier()

#set with max_sample at .5 nd max feature at .5
baggingKNeighbor = BaggingClassifier(
                KClass)

baggingRForest = BaggingClassifier(
                RClass)

baggingDTree = BaggingClassifier(
                DClass)

#using cross_valication score
from sklearn.model_selection import cross_val_score
#bagging the iris dataset

baggingIrisKNeighborIrisScore = cross_val_score(baggingKNeighbor, Iris_X, Iris_y)
baggingIrisRForestIrisScore = cross_val_score(baggingRForest, Iris_X, Iris_y)
baggingIrisDTreeScore = cross_val_score(baggingDTree, Iris_X, Iris_y)

baggingIosphereKNeighborScore = cross_val_score(baggingKNeighbor, IosphereX, IosphereY)
baggingIosphereRForestScore = cross_val_score(baggingRForest, IosphereX, IosphereY)
baggingIosphereDTreeScore = cross_val_score(baggingDTree, IosphereX, IosphereY)

#print bagging score mean


print(baggingIrisKNeighborIrisScore.mean())
print(baggingIrisRForestIrisScore.mean())
print(baggingIrisDTreeScore.mean())

print(baggingIosphereKNeighborScore.mean())
print(baggingIosphereRForestScore.mean())
print(baggingIosphereDTreeScore.mean())


