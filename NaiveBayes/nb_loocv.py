import numpy as np
import pandas as pd 
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.cross_validation import LeaveOneOut, cross_val_score
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
        loo = LeaveOneOut(len(y))
        correct_1 = 0
        wrong_1 = 0
        correct_0 = 0
        wrong = 0
        for train, test in loo:
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            self.clf.fit(X_train, y_train)
            predict = self.clf.predict(X_test)
            cnf_matrix_mnb = confusion_matrix(y_test, predict)
            #print()
            #print("predicted %s" % predict)
            #print("original %s" % y_test)
            if (predict == 1 and y_test ==1):
                correct_1 = correct_1 + 1
            elif(predict == 0 and y_test ==0):
                correct_0 = correct_0 + 1
            else:
                wrong = wrong + 1
        print()
        print("correct_1 %s" %correct_1)
        print("correct_0 %s" %correct_0)
        print("wrong %s" %wrong)

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

