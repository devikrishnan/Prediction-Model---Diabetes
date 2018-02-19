from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import LeaveOneOut, cross_val_score
from sklearn import preprocessing
from sklearn.cross_validation import KFold

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
        loo = LeaveOneOut(len(y))
        correct_1 = 0
        correct_0 = 0
        wrong = 0
        for train, test in loo:
        	X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        	clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)
        	clf.fit(X_train, y_train)
        	predict = clf.predict(X_test)
        	cnf_matrix_mnb = confusion_matrix(y_test, predict)
        	if (predict == 1 and y_test ==1):
        		correct_1 = correct_1 + 1
        	elif(predict == 0 and y_test == 0):
        		correct_0 = correct_0 + 1
        	else:
        		wrong = wrong + 1
        print()
        print("correct_1 %s" %correct_1)
        print("correct_0 %s" %correct_0)
        print("wrong %s" %wrong)


        	#print(cnf_matrix_mnb)

svm().fit()

	
#cnf_matrix_mnb = confusion_matrix(y_test, predict)
#print(P)
'''print(predict)
print(cnf_matrix_mnb)


clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
estimator = clf.fit(X, Y)
cv = LeaveOneOut(len(Y))
scores = cross_val_score(estimator, X, Y, cv = cv)
print(scores)
cnf_matrix_mnb = confusion_matrix(Y, scores)
print('Best score for data1:', clf.best_score_)
print(cnf_matrix_mnb)'''
#print("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))

#print(X1)




