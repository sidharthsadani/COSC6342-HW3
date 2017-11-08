from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

with open("iosphere_data.txt") as f:
    contents = f.readlines()

contents = [x.strip('\n').split(',') for x in contents] 

#for content in contents:
    #print (content)
        
print (contents)
