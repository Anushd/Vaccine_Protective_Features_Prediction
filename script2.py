from xlrd import open_workbook
from sklearn import svm
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoLarsCV, RandomizedLasso, MultiTaskLasso
from sklearn import tree, datasets, preprocessing
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt
import time

data1 = open_workbook('/Users/anush/Desktop/data1.xlsx').sheet_by_index(0)
data0 = open_workbook('/Users/anush/Desktop/data0.xlsx').sheet_by_index(0)

g1 = []
g0 = []
for i in range(1,data1.nrows):
	tmp1 = np.array(data1.row_values(i)[1:81])
	g1.append(tmp1)
for i in range(1,data0.nrows):
	tmp0 = np.array(data0.row_values(i)[1:81])
	g0.append(tmp0)
	
data = np.concatenate((g1,g0))
x = np.array(data)
x = stats.zscore(x, axis=0)
y = np.array(np.concatenate((np.ones(len(g1)),np.zeros(len(g0)))))

model = LassoLarsCV(cv=20).fit(x, y)
clf = Lasso(alpha=model.alpha_)
clf.fit(x,y)

print clf.coef_

sfm = SelectFromModel(clf, threshold=0.05)
sfm.fit(x,y)
x = sfm.transform(x)

clf = svm.SVC(kernel='rbf')
#clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
#clf = RandomForestClassifier(n_estimators=100)
#bdt = AdaBoostClassifier(clf, algorithm="SAMME", n_estimators=500)
scores = cross_val_score(clf, x, y, cv=20)

print scores.mean(), "+/-", scores.std()

'''kf = KFold(n_splits=20)
for train_index, test_index in kf.split(x):
	x_train, x_test = x[train_index], x[test_index]
	y_train, y_test = y[train_index], y[test_index]
	clf.fit(x_train,y_train)
	pred = clf.predict(x_test)	
	print pred
	print y_test
	print'''