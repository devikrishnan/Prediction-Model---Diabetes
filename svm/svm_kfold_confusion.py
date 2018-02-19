import numpy as np
import pandas as pd 
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

parameter_candidates = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        ]

class svm():

    def fit(self):
        df = pd.read_csv('Datasetnew.csv',header=None)
        h=np.asarray(df)
        dataset = np.nan_to_num(h)
        XX = dataset[:,1:65]
        y = dataset[:,0]
        X = preprocessing.normalize(XX)
        clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)
        k_fold = KFold(10)
        for k, (train, test) in enumerate(k_fold.split(X, y)):
            clf.fit(X[train], y[train])
            predict = clf.predict(X[test])
            cnf_matrix_mnb = confusion_matrix(y[test], predict)
            print()
            print(y[train])
            #print()
            print(clf.score(X[test], y[test]))
            print(cnf_matrix_mnb)

svm().fit()




