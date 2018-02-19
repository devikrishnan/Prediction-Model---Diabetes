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
        scores = ['precision', 'recall']
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
            for k, (train, test) in enumerate(k_fold.split(X, y)):
                self.clf.fit(X[train], y[train])
                print("Detailed classification report:")
                print()
                y_true, y_pred = y[test], self.clf.predict(X[test])
                print(classification_report(y_true, y_pred))
                print()
                print("Detailed confusion matrix:")
                cnf_matrix_mnb = confusion_matrix(y[test], y_pred)
                print(self.clf.score(X[test], y[test]))
                print(cnf_matrix_mnb)
                print()
                print()


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
