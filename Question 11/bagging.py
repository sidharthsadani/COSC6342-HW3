from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

contents = []
with open("iosphere_data.txt") as f:
    for line in f:
        contents.append([s for s in line.strip().split(',')])

X = [[float(p) for p in ex[:-1]] for ex in contents]
Y = [1.0 if sublist[-1]=='g' else 0.0 for sublist in contents]
print([s[-1] for s in contents])
print(X)
print(Y)