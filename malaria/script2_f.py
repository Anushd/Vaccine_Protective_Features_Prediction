from xlrd import open_workbook
from sklearn import svm
from sklearn.svm import SVC
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoLarsCV, LassoCV
from sklearn import tree, datasets, preprocessing
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from statistics import mean

#import data files
data1 = open_workbook('/Users/anush/Desktop/data1.xlsx').sheet_by_index(0)
data0 = open_workbook('/Users/anush/Desktop/data0.xlsx').sheet_by_index(0)

#append values to arrays
g1 = []
g0 = []
for i in range(1,data1.nrows):
	tmp1 = np.array(data1.row_values(i)[1:81])
	g1.append(tmp1)
for i in range(1,data0.nrows):
	tmp0 = np.array(data0.row_values(i)[1:81])
	g0.append(tmp0)

#combine arrays across classes
x_0 = np.concatenate((g1,g0))
x_0 = np.array(x_0)

#take z-scores 
x_0 = stats.zscore(x_0, axis=0)

#create labels for data set (1/0)
y_0 = np.array(np.concatenate((np.ones(len(g1)),np.zeros(len(g0)))))

scores=[]

kf1 = KFold(n_splits=20)
for train_index1, test_index1 in kf1.split(x_0):
	
	x1, x_test1 = x_0[train_index1], x_0[test_index1]
	y1, y_test1 = y_0[train_index1], y_0[test_index1]
	
	features_select = [2, 43, 55, 64]
	
	x1 = x1[:,features_select]
	x_test1 = x_test1[:,features_select]
	
	#generate svm model for classification 
	#gamma_list = dict(gamma= [i*0.0001 for i in range(5, 1250, 5)])
	#clf1 = GridSearchCV(estimator=SVC(), param_grid=gamma_list)
	#clf1.fit(x1, y1)
	clf = svm.SVC(kernel='poly', degree=3)#, C=clf1.best_estimator_.C)
	clf.fit(x1,y1)
	y_pred = clf.predict(x_test1)
	score = accuracy_score(y_test1, y_pred)
	scores.append(score) 

print mean(scores), '+/-', np.std(scores)

	#clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
	#clf = RandomForestClassifier(n_estimators=100)
	#bdt = AdaBoostClassifier(clf, algorithm="SAMME", n_estimators=500)

	#stratified k-fold cross validation
	#scores = cross_val_score(clf, x, y, cv=20)

	#print scores.mean(), "+/-", scores.std()

'''kf = KFold(n_splits=20)
for train_index, test_index in kf.split(x):
	x_train, x_test = x[train_index], x[test_index]
	y_train, y_test = y[train_index], y[test_index]
	clf.fit(x_train,y_train)
	pred = clf.predict(x_test)	
	print pred
	print y_test
	print'''
