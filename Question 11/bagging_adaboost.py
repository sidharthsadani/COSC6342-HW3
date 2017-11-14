
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
#Sidharth Sadani ppsid 1503352

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets

#matplot library from example in scikit learn
import numpy as np

#using this for the other data set needed -- scikit leanr already included dataset
import sklearn.datasets as data_sets

#make meshgrid method from http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

#using the iris dataset from scikit learn
Iris_X, Iris_y = data_sets.load_iris(return_X_y=True)
iris = data_sets.load_iris()

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
EClass = ExtraTreesClassifier()
GClass = GaussianProcessClassifier()

#set bagging and adaboost classifier with base classifier of Kneighbor, RandomForest and Decision Trees

baggingKNeighbor = BaggingClassifier(KClass)
baggingRForest = BaggingClassifier(RClass)
baggingDTree = BaggingClassifier(DClass)

adaBoostEClass = AdaBoostClassifier(EClass)
adaBoostRForest = AdaBoostClassifier(RClass)
adaBoostDTree = AdaBoostClassifier(DClass)

#using cross_valication score
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import r2_score as r2

# irisBaggingKNeighbor = baggingKNeighbor.fit(Iris_X, Iris_y)
# irisBaggingRForest = baggingRForest.fit(Iris_X, Iris_y)
# irisBaggingDTree = baggingDTree.fit(Iris_X, Iris_y)
#
# iosphereBaggingKNeighbor = baggingKNeighbor.fit(IosphereX, IosphereY)
# iosphereBaggingRForest = baggingRForest.fit(IosphereX, IosphereY)
# iosphereBaggingDTree = baggingDTree.fit(IosphereX, IosphereY)
#
# irisAdaETree = adaBoostEClass.fit(Iris_X, Iris_y)
# irisAdaRForest = adaBoostRForest.fit(Iris_X, Iris_y)
# irisAdaDTree = adaBoostDTree.fit(Iris_X, Iris_y)
#
# iosphereAdaETree = adaBoostEClass.fit(IosphereX, IosphereY)
# iosphereAdaRForest = adaBoostRForest.fit(IosphereX, IosphereY)
# iosphereAdaDTree = adaBoostDTree.fit(IosphereX, IosphereY)
#
# models = (irisBaggingKNeighbor, irisBaggingRForest, irisBaggingDTree, irisAdaETree,irisAdaRForest, irisAdaDTree)
# titles = ('irisBaggingKNeighbor', 'irisBaggingRForest', 'irisBaggingDTree', 'irisAdaETree', 'irisAdaRForest', 'irisAdaDTree')

# models = (irisBaggingKNeighbor, irisBaggingRForest, irisAdaDTree, iosphereBaggingKNeighbor, iosphereBaggingRForest,
#           iosphereBaggingDTree, irisAdaETree,irisAdaRForest, irisAdaDTree, iosphereAdaETree, iosphereAdaRForest,
#           iosphereAdaDTree)
# titles = ('irisBaggingKNeighbor', 'irisBaggingRForest', 'irisBaggingDTree', 'iosphereBaggingKNeighbor',
#           'iosphereBaggingRForest',   'iosphereBaggingDTree', 'irisAdaETree', 'irisAdaRForest', 'irisAdaDTree',
#           'iosphereAdaETree', 'iosphereAdaRForest',  'iosphereAdaDTree')
#
# plt.figure(figsize=(8, 8))
# plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
# plt.subplot(321)
#
# plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
#             s=25, edgecolor='k')
#
# plt.show()

# baggingKNeighbor = BaggingClassifier(KClass)
# baggingRForest = BaggingClassifier(RClass)
# baggingDTree = BaggingClassifier(DClass)
#
# adaBoostEClass = AdaBoostClassifier(EClass)
# adaBoostRForest = AdaBoostClassifier(RClass)
# adaBoostDTree = AdaBoostClassifier(DClass)


for i in range(2,11,2):
    print('==============================================')
    print('with Kfold of', i)
    print('==============================================')

    #using kfold validation and splitting

    kf = KFold(n_splits=i)

    #bagging the iris dataset

    print ("Iris dataset")

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(Iris_X):
        Xtrain , Xcrosset = Iris_X[train_index], Iris_X[cross_index]
        ytrain , ycrosset = Iris_y[train_index], Iris_y[cross_index]
        baggingKNeighbor.fit(Xtrain, ytrain)
        train_error += baggingKNeighbor.score(Xtrain, ytrain) # train error
        cv_error += baggingKNeighbor.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i

    print("baggingKNeighbor train error,", train_error,  ",baggingKNeighbor cv error,", cv_error)

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(Iris_X):
        Xtrain , Xcrosset = Iris_X[train_index], Iris_X[cross_index]
        ytrain , ycrosset = Iris_y[train_index], Iris_y[cross_index]
        baggingRForest.fit(Xtrain, ytrain)
        train_error += baggingRForest.score(Xtrain, ytrain) # train error
        cv_error += baggingRForest.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i
    print("baggingRForest train error,", train_error, ",baggingRForest cv error,", cv_error)

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(Iris_X):
        Xtrain , Xcrosset  = Iris_X[train_index], Iris_X[cross_index]
        ytrain , ycrosset = Iris_y[train_index], Iris_y[cross_index]
        baggingDTree.fit(Xtrain, ytrain)
        train_error += baggingDTree.score(Xtrain, ytrain) # train error
        cv_error += baggingDTree.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i

    print("baggingDTree train error,", train_error, ",baggingDTree cv error,", cv_error)

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(Iris_X):
        Xtrain , Xcrosset = Iris_X[train_index], Iris_X[cross_index]
        ytrain , ycrosset = Iris_y[train_index], Iris_y[cross_index]
        adaBoostEClass.fit(Xtrain, ytrain)
        train_error += adaBoostEClass.score(Xtrain, ytrain) # train error
        cv_error += adaBoostEClass.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i

    print("adaBoostEClass train error,", train_error,  ",adaBoostEClass cv error,", cv_error)

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(Iris_X):
        Xtrain , Xcrosset = Iris_X[train_index], Iris_X[cross_index]
        ytrain , ycrosset = Iris_y[train_index], Iris_y[cross_index]
        adaBoostRForest.fit(Xtrain, ytrain)
        train_error += adaBoostRForest.score(Xtrain, ytrain) # train error
        cv_error += adaBoostRForest.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i
    print("adaBoostRForest train error,", train_error, ",adaBoostRForest cv error,", cv_error)

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(Iris_X):
        Xtrain , Xcrosset  = Iris_X[train_index], Iris_X[cross_index]
        ytrain , ycrosset = Iris_y[train_index], Iris_y[cross_index]
        adaBoostDTree.fit(Xtrain, ytrain)
        train_error += adaBoostDTree.score(Xtrain, ytrain) # train error
        cv_error += adaBoostDTree.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i

    print("adaBoostDTree train error,", train_error, ",adaBoostDTree cv error,", cv_error)

    print ("Iosphere dataset")

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(IosphereX):
        Xtrain , Xcrosset = IosphereX[train_index], IosphereX[cross_index]
        ytrain , ycrosset = IosphereY[train_index], IosphereY[cross_index]
        baggingKNeighbor.fit(Xtrain, ytrain)
        train_error += baggingKNeighbor.score(Xtrain, ytrain) # train error
        cv_error += baggingKNeighbor.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i

    print("baggingKNeighbor train error,", train_error,  ",baggingKNeighbor cv error,", cv_error)

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(IosphereX):
        Xtrain , Xcrosset = IosphereX[train_index], IosphereX[cross_index]
        ytrain , ycrosset = IosphereY[train_index], IosphereY[cross_index]
        baggingRForest.fit(Xtrain, ytrain)
        train_error += baggingRForest.score(Xtrain, ytrain) # train error
        cv_error += baggingRForest.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i
    print("baggingRForest train error,", train_error, ",baggingRForest cv error,", cv_error)

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(IosphereX):
        Xtrain , Xcrosset  = IosphereX[train_index], IosphereX[cross_index]
        ytrain , ycrosset = IosphereY[train_index], IosphereY[cross_index]
        baggingDTree.fit(Xtrain, ytrain)
        train_error += baggingDTree.score(Xtrain, ytrain) # train error
        cv_error += baggingDTree.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i

    print("baggingDTree train error,", train_error, ",baggingDTree cv error,", cv_error)

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(IosphereX):
        Xtrain , Xcrosset = IosphereX[train_index], IosphereX[cross_index]
        ytrain , ycrosset = IosphereY[train_index], IosphereY[cross_index]
        adaBoostEClass.fit(Xtrain, ytrain)
        train_error += adaBoostEClass.score(Xtrain, ytrain) # train error
        cv_error += adaBoostEClass.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i

    print("adaBoostEClass train error,", train_error,  ",adaBoostEClass cv error,", cv_error)

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(IosphereX):
        Xtrain , Xcrosset = IosphereX[train_index], IosphereX[cross_index]
        ytrain , ycrosset = IosphereY[train_index], IosphereY[cross_index]
        adaBoostRForest.fit(Xtrain, ytrain)
        train_error += adaBoostRForest.score(Xtrain, ytrain) # train error
        cv_error += adaBoostRForest.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i
    print("adaBoostRForest train error,", train_error, ",adaBoostRForest cv error,", cv_error)

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(IosphereX):
        Xtrain , Xcrosset  = IosphereX[train_index], IosphereX[cross_index]
        ytrain , ycrosset = IosphereY[train_index], IosphereY[cross_index]
        adaBoostDTree.fit(Xtrain, ytrain)
        train_error += adaBoostDTree.score(Xtrain, ytrain) # train error
        cv_error += adaBoostDTree.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i

    print("adaBoostDTree train error,", train_error, ",adaBoostDTree cv error,", cv_error)
    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(IosphereX):
        Xtrain , Xcrosset = IosphereX[train_index], IosphereX[cross_index]
        ytrain , ycrosset = IosphereY[train_index], IosphereY[cross_index]
        baggingKNeighbor.fit(Xtrain, ytrain)
        train_error += baggingKNeighbor.score(Xtrain, ytrain) # train error
        cv_error += baggingKNeighbor.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i

    print("baggingKNeighbor train error,", train_error,  ",baggingKNeighbor cv error,", cv_error)

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(IosphereX):
        Xtrain , Xcrosset = IosphereX[train_index], IosphereX[cross_index]
        ytrain , ycrosset = IosphereY[train_index], IosphereY[cross_index]
        baggingRForest.fit(Xtrain, ytrain)
        train_error += baggingRForest.score(Xtrain, ytrain) # train error
        cv_error += baggingRForest.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i
    print("baggingRForest train error,", train_error, ",baggingRForest cv error,", cv_error)

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(IosphereX):
        Xtrain , Xcrosset  = IosphereX[train_index], IosphereX[cross_index]
        ytrain , ycrosset = IosphereY[train_index], IosphereY[cross_index]
        baggingDTree.fit(Xtrain, ytrain)
        train_error += baggingDTree.score(Xtrain, ytrain) # train error
        cv_error += baggingDTree.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i

    print("baggingDTree train error,", train_error, ",baggingDTree cv error,", cv_error)

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(IosphereX):
        Xtrain , Xcrosset = IosphereX[train_index], IosphereX[cross_index]
        ytrain , ycrosset = IosphereY[train_index], IosphereY[cross_index]
        adaBoostEClass.fit(Xtrain, ytrain)
        train_error += adaBoostEClass.score(Xtrain, ytrain) # train error
        cv_error += adaBoostEClass.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i

    print("adaBoostEClass train error,", train_error,  ",adaBoostEClass cv error,", cv_error)

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(IosphereX):
        Xtrain , Xcrosset = IosphereX[train_index], IosphereX[cross_index]
        ytrain , ycrosset = IosphereY[train_index], IosphereY[cross_index]
        adaBoostRForest.fit(Xtrain, ytrain)
        train_error += adaBoostRForest.score(Xtrain, ytrain) # train error
        cv_error += adaBoostRForest.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i
    print("adaBoostRForest train error,", train_error, ",adaBoostRForest cv error,", cv_error)

    train_error, cv_error = 0.0, 0.0
    for train_index, cross_index in kf.split(IosphereX):
        Xtrain , Xcrosset  = IosphereX[train_index], IosphereX[cross_index]
        ytrain , ycrosset = IosphereY[train_index], IosphereY[cross_index]
        adaBoostDTree.fit(Xtrain, ytrain)
        train_error += adaBoostDTree.score(Xtrain, ytrain) # train error
        cv_error += adaBoostDTree.score(Xcrosset, ycrosset) # cv error

    train_error /= i
    cv_error /= i

    print("adaBoostDTree train error,", train_error, ",adaBoostDTree cv error,", cv_error)









    #
    # baggingIrisKNeighborIrisScore = cross_val_score(baggingKNeighbor, Iris_X, Iris_y, cv=kf)
    # baggingIrisRForestIrisScore = cross_val_score(baggingRForest, Iris_X, Iris_y, cv=kf)
    # baggingIrisDTreeScore = cross_val_score(baggingDTree, Iris_X, Iris_y, cv=kf)
    #
    # baggingIrisKNeighborIrisPredict = r2(Iris_y, cross_val_predict(baggingKNeighbor, Iris_X, Iris_y, cv=kf))
    # baggingIrisRForestIrisPredict = r2(Iris_y, cross_val_predict(baggingRForest, Iris_X, Iris_y, cv=kf))
    # baggingIrisDTreePredict = r2(Iris_y, cross_val_predict(baggingDTree, Iris_X, Iris_y, cv=kf))
    #
    #
    # #bagging iosphere dataset
    # baggingIosphereKNeighborScore = cross_val_score(baggingKNeighbor, IosphereX, IosphereY, cv=kf)
    # baggingIosphereRForestScore = cross_val_score(baggingRForest, IosphereX, IosphereY, cv=kf)
    # baggingIosphereDTreeScore = cross_val_score(baggingDTree, IosphereX, IosphereY, cv=kf)
    #
    # baggingIosphereKNeighborPredict = r2(IosphereY, cross_val_predict(baggingKNeighbor, IosphereX, IosphereY, cv=kf))
    # baggingIosphereRForestPredict = r2(IosphereY, cross_val_predict(baggingRForest, IosphereX, IosphereY, cv=kf))
    # baggingIosphereDTreePredict = r2(IosphereY, cross_val_predict(baggingDTree, IosphereX, IosphereY, cv=kf))

    #print bagging score mean
    # print('========================')
    #
    # print('Bagging mean score')
    #
    # print('========================')
    #
    # print('Average score for bagging using KNeighbor - Iris:', baggingIrisKNeighborIrisScore.mean())
    # print('Average score for bagging using Random Forest - Iris:', baggingIrisRForestIrisScore.mean())
    # print('Average score for bagging using Decision Tree - Iris:', baggingIrisDTreeScore.mean())
    #
    # print()
    #
    # print('Average score for bagging using KNeighbor - Iosphere:', baggingIosphereKNeighborScore.mean())
    # print('Average score for bagging using Random Forest - Iosphere:', baggingIosphereRForestScore.mean())
    # print('Average score for bagging using Decision Tree - Iosphere:',  baggingIosphereDTreeScore.mean())
    #
    # print('Average score for bagging using KNeighbor - Iosphere:', baggingIosphereKNeighborScore.mean())
    # print('Average score for bagging using Random Forest - Iosphere:', baggingIosphereRForestScore.mean())
    # print('Average score for bagging using Decision Tree - Iosphere:',  baggingIosphereDTreeScore.mean())

    #print bagging prediction
    # print()
    #
    # print('========================')
    #
    # print('Bagging prediction accuracy')
    #
    # print('========================')
    #
    # print('Prediction accuracy for bagging using KNeighbor - Iris:', baggingIrisKNeighborIrisPredict)
    # print('Prediction accuracy for bagging using Random Forest - Iris:', baggingIrisRForestIrisPredict)
    # print('Prediction accuracy for bagging using Decision Tree - Iris:', baggingIrisDTreePredict)
    #
    # print()
    #
    # print('Prediction accuracy for bagging using KNeighbor - Iosphere:', baggingIosphereKNeighborPredict)
    # print('Prediction accuracy for bagging using Random Forest - Iosphere:', baggingIosphereRForestPredict)
    # print('Prediction accuracy for bagging using Decision Tree - Iosphere:',  baggingIosphereDTreePredict)
    #
    # print()


    #adaBoost the iris dataset

    # adaBoostIrisETreeIrisScore = cross_val_score(adaBoostEClass, Iris_X, Iris_y, cv=kf)
    # adaBoostIrisRForestIrisScore = cross_val_score(adaBoostRForest, Iris_X, Iris_y, cv=kf)
    # adaBoostIrisDTreeScore = cross_val_score(adaBoostDTree, Iris_X, Iris_y, cv=kf)
    #
    # adaBoostIrisETreeIrisPredict = r2(Iris_y, cross_val_predict(adaBoostEClass, Iris_X, Iris_y, cv=kf))
    # adaBoostIrisRForestIrisPredict = r2(Iris_y, cross_val_predict(adaBoostRForest, Iris_X, Iris_y, cv=kf))
    # adaBoostIrisDTreePredict = r2(Iris_y, cross_val_predict(adaBoostDTree, Iris_X, Iris_y, cv=kf))
    #
    # #adaBoost iosphere dataset
    # adaBoostIosphereETreeScore = cross_val_score(adaBoostEClass, IosphereX, IosphereY, cv=kf)
    # adaBoostIosphereRForestScore = cross_val_score(adaBoostRForest, IosphereX, IosphereY, cv=kf)
    # adaBoostIosphereDTreeScore = cross_val_score(adaBoostDTree, IosphereX, IosphereY, cv=kf)
    #
    # adaBoostIosphereETreePredict = r2(IosphereY, cross_val_predict(adaBoostEClass, IosphereX, IosphereY, cv=kf))
    # adaBoostIosphereRForestPredict = r2(IosphereY, cross_val_predict(adaBoostRForest, IosphereX, IosphereY, cv=kf))
    # adaBoostIosphereDTreePredict = r2(IosphereY, cross_val_predict(adaBoostDTree, IosphereX, IosphereY, cv=kf))

    #print adaBoost score mean
    # print('========================')
    #
    # print('AdaBoost mean score')
    #
    # print('========================')
    #
    # print('Average score for adaBoost using Extra Tree - Iris:', adaBoostIrisETreeIrisScore.mean())
    # print('Average score for adaBoost using Random Forest - Iris:', adaBoostIrisRForestIrisScore.mean())
    # print('Average score for adaBoost using Decision Tree - Iris:', adaBoostIrisDTreeScore.mean())
    #
    # print()
    #
    # print('Average score for adaBoost using Extra Tree - Iosphere:', adaBoostIosphereETreeScore.mean())
    # print('Average score for adaBoost using Random Forest - Iosphere:', adaBoostIosphereRForestScore.mean())
    # print('Average score for adaBoost using Decision Tree - Iosphere:',  adaBoostIosphereDTreeScore.mean())
    #
    # print()

    #print adaBoost prediction
    # print('========================')
    #
    # print('AdaBoost prediction accuracy')
    #
    # print('========================')
    #
    # print('Prediction accuracy for adaBoost using Extra Tree - Iris:', adaBoostIrisETreeIrisPredict)
    # print('Prediction accuracy for adaBoost using Random Forest - Iris:', adaBoostIrisRForestIrisPredict)
    # print('Prediction accuracy for adaBoost using Decision Tree - Iris:', adaBoostIrisDTreePredict)
    #
    # print()
    #
    # print('Prediction accuracy for adaBoost using Extra Tree - Iosphere:', adaBoostIosphereETreePredict)
    # print('Prediction accuracy for adaBoost using Random Forest - Iosphere:', adaBoostIosphereRForestPredict)
    # print('Prediction accuracy for adaBoost using Decision Tree - Iosphere:',  adaBoostIosphereDTreePredict)
    #
    # print()






