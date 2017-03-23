#Import dependencies
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
# Read the training data file
df = pd.read_csv('Data/train.csv', low_memory = False)
#df_test = pd.read_csv('Data/test.csv', header = None, low_memory = False)
y = df.pop('label')

X_train, X_test, y_train, y_test =  train_test_split(df, y, test_size=0.2, stratify = y)
print (y_test)
# call the KNN classifier using n as the nearest odd number to the sqrt
# of the total number of samples
def fit(X,Y,x,y):
    m = int(np.ceil(np.sqrt(len(X))))
    if m%2==0:
        n=m-1
    else:
        n=m
    error_=[]
    for i in range(n-20, n+20, 2):
        error=0
        clf = KNeighborsClassifier(n_neighbors=i, weights='uniform')
        clf.fit(X, Y)
        error_.append(clf.score(x,y))
    return error_
y=fit(X_train, y_train, X_test, y_test)
print(y)

def predict(X,Y,x,y):
    clf = KNeighborsClassifier(n_neighbors=99, weights='distance')
    clf.fit(X, Y)
    print('Actual: ', y)
    print('Prediction: ', clf.predict(x))
    print ('Accuracy: ', clf.score(x,y))

predict(X_train, y_train, X_test, y_test)
