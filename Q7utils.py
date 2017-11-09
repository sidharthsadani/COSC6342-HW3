import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA, NMF


print("Hello World")

def GetBestFeatIris(numFeat = 2):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    
    pca = PCA(n_components = numFeat)
    X_r = pca.fit(X).transform(X)


    print(numFeat)

GetBestFeatIris(2)
