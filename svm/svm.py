from __future__ import print_function
import pandas as pd 
import numpy as np 
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

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
        k_fold = KFold(10)
        scores = ['precision', 'recall']
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
            clf = GridSearchCV(SVC(), parameter_candidates, cv=5, scoring='%s_macro' % score)
            for k, (train, test) in enumerate(k_fold.split(X, y)):
                clf.fit(X[train], y[train])
                #print()
                #print(y[train])
                #print()
                print("Best parameters set found on development set:")
                print()
                print(clf.best_params_)
                print()
                '''print("Grid scores on development set:")
                print()
                means = clf.cv_results_['mean_test_score']
                stds = clf.cv_results_['std_test_score']
                for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
                print()'''
                print("Detailed classification report:")
                print()
                y_true, y_pred = y[test], clf.predict(X[test])
                cnf_matrix_mnb = confusion_matrix(y[test], y_pred)
                print(classification_report(y_true, y_pred))
                print()
                print(clf.score(X[test], y[test]))
                print(cnf_matrix_mnb)
                print()
                print()

svm().fit()
