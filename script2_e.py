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

#svm scores for each outer fold
scores=[]

#outer folds
nfold_outer=10
kf1 = KFold(n_splits=nfold_outer)

#count of outer fold
count_3=0

#array of selected features for each outer fold
array1 = np.zeros((nfold_outer,80))

#Count of folds which had greater than 0 features selected
fold_count=0

for train_index1, test_index1 in kf1.split(x_0):
	
	#partition data by outer k-fold
	x1, x_test1 = x_0[train_index1], x_0[test_index1]
	y1, y_test1 = y_0[train_index1], y_0[test_index1]
	
	#Least Angle Regression for optimal alpha selection
	model = LassoLarsCV(cv=20).fit(x1, y1)
	
	#inner fold
	nfold_inner = 5
	
	#array of selected features for each inner fold
	features=np.zeros((nfold_inner,80))
		
	#count of inner fold
	count1=0
	
	#inner k-fold
	kf2 = KFold(n_splits=nfold_inner)
	
	#count of folds satisfying classification accuracy threshhold
	thresh_count=0
	
	for train_index2, test_index2 in kf2.split(x1):
		
		#partition data by inner k-fold
		x2, x_test2 = x1[train_index2], x1[test_index2]
		y2, y_test2 = y1[train_index2], y1[test_index2]
		
		# lasso model
		clf = Lasso(alpha=model.alpha_)
		clf.fit(x2,y2)
		
		# select non-zero features from lasso and apply to data
		sfm = SelectFromModel(clf, threshold=0.0005)
		sfm.fit(x2,y2)
		x2 = sfm.transform(x2)
		x_test2 = sfm.transform(x_test2)
		
		# svm model
		clf2 = svm.SVC(kernel='poly')
		clf2.fit(x2,y2)
		
		# svm score
		pred = clf2.predict(x_test2)
		score = accuracy_score(y_test2, pred)
		#print score
		
		# include only selected features that satisfy svm score threshold (50% accuracy)
		if score>=0.5:
			thresh_count+=1
			count2=0
			for i in clf.coef_:
				if abs(i)>0:
					features[count1,count2]=1
				count2+=1
		count1+=1
	
	#array of features selected from inner cv	
	features_select=[]
	
	for i in range(80):
		feature_count=0
		
		# count number of times each feature selected across all inner folds
		for j in range(nfold_inner):
			if features[j,i]>0:
				feature_count+=1
		
		# select feature if number of times selected satisfies threshhold (60% of thresh_count)
		if feature_count>thresh_count*0.7:
			features_select.append(i)
	
	# apply selected features to outer fold data
	x1 = x1[:,features_select]
	x_test1 = x_test1[:,features_select]
	
	# add selected features to array1	
	for i in features_select:
		array1[count_3,i]=1
	#print x1.shape
	
	# svm model
	clf = svm.SVC(kernel='poly', degree=3)
	
	# calculate score for svm model only if >0 features selected by inner cv
	if x1.shape[1]!=0:
		fold_count += 1
		clf.fit(x1,y1)
		y_pred = clf.predict(x_test1)
		score = accuracy_score(y_test1, y_pred)
		scores.append(score) 
	
	count_3+=1

print mean(scores), '+/-', np.std(scores)
#print scores

#final features
features_final=[]

#include features selected >=7 times across outer folds
for i in range(80):
	feature_count=0
	for j in range(nfold_outer):
		if array1[j,i]>0:
			feature_count+=1
		
	if feature_count>=6:
		features_final.append(i)
print features_final
