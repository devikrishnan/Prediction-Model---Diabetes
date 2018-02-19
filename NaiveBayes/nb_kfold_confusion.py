import numpy as np
import pandas as pd 
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import classification_report


class NB():
    def __init__(self, clf):
        self.clf = clf

    def fit(self):
        df = pd.read_csv('Datasetnew.csv',header=None)
        h=np.asarray(df)
        dataset = np.nan_to_num(h)
        XX = dataset[:,1:65]
        y = dataset[:,0]
        X = preprocessing.normalize(XX)
        k_fold = KFold(10)
        for k, (train, test) in enumerate(k_fold.split(X, y)):
            self.clf.fit(X[train], y[train])
            predict = self.clf.predict(X[test])
            print()
            #print(y[train])
            #print()
            cnf_matrix_mnb = confusion_matrix(y[test], predict)
            print(self.clf.score(X[test], y[test]))
            print(cnf_matrix_mnb)


print()
print("Naive Bayes: MultinomialNB()")
print()
NB(MultinomialNB()).fit()

print()
print("Naive Bayes: GaussianNB()")
print()
NB(GaussianNB()).fit()

print()
print("Naive Bayes: BernoulliNB()")
print()
NB(BernoulliNB()).fit()

