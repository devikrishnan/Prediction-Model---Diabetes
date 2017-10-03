import pandas as pd
import numpy as np
from sklearn import svm


data_train = pd.read_csv("apnea-ihr-spo2-train-data.csv", header= None )
data_test = pd.read_csv("apnea-ihr-spo2-test-data.csv", header= None)

dtrain = np.array(data_train, dtype=np.float)
dtest = np.array(data_test, dtype=np.float)
X = dtrain[:,1:4]
Y = dtrain[:,0]
#print X
#print Y

clf = svm.SVC()
clf.fit(X,Y)
print clf
print clf.predict([[61.,86.,0.709302326]]) 

#print dtrain
#print dtest


