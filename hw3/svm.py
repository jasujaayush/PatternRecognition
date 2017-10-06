import csv
import sys
import urllib
import random
import numpy as np
import pandas as pd  
from math import exp
from math import log
import seaborn as sns                                                
import scipy.optimize
import matplotlib as mpl        
from random import shuffle                      
from sklearn.svm import SVC                 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score


def parseData(fname):
	with open(fname, 'rU') as f:
		reader = csv.reader(f)
		data = list(list(rec) for rec in csv.reader(f, delimiter=','))
		mdata = []
		feat = []
		for row  in data[1:]:
			feat = [int(e) for e in row]
			mdata.append(feat)
		shuffle(mdata)
	return mdata	

def subset(p, data):
	split = int(.01*p*len(data))
	s = data[:split]
	return s

def fsplit(f, data):
	t = int(len(data)/float(f))
	sets = []
	for i in range(f):
		temp = data[i*t : t*(i+1)]
		sets.append(temp)
	return sets

def getftlbls(trsubset):
	y = [r[0] for r in trsubset]
	x = [r[1:] for r in trsubset]
	yset = set(y)
	for v in yset:
		count = y.count(v)
		if count < 3:
			for i in range(count):
				index = y.index(v)
				del y[index]
				del x[index]
	return x, y


temp1 = parseData('covtype.data')
t = int(.50*len(temp1))  #take 50 percent of train data
temp1 = temp1[:t]
temp = [[d[-1]] + d[:-1] for d in temp1]
del temp1
x = int(.75*len(temp))
testdata = temp[x:]
traindata = temp[:x]
#temp = parseData('train.csv')	

pca = PCA(n_components=15)   
y = [r[0] for r in traindata] 
x = [r[1:] for r in traindata] 
x_pca = pca.fit_transform(x)     
x_inv_pca = pca.inverse_transform(x_pca) 
loss = ((x - x_inv_pca) ** 2).mean()
print loss
traindata = [[y[i]] + x_pca[i].tolist() for i in range(len(x_pca))] 
yt = [r[0] for r in testdata] 
xt = [r[1:] for r in testdata]
xt = pca.transform(xt)
testdata = [[y[i]] + x_pca[i].tolist() for i in range(len(xt))] 

C_range = np.logspace(-6, -2, 6)
gamma_range = np.logspace(-9, -5, 4)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
bestg = []
bestc = []
for dpercent in [50, 75, 100]:
	dsub = subset(dpercent, traindata)
	numsets = len(dsub)/5000
	print numsets
	sets = fsplit(numsets, dsub)
	print "Set Size : ", len(sets[0])
	for trkernel in ['linear', 'rbf']:
		print "Processing kernel : ", trkernel
		tempg = []
		tempc = []
		count = 0
		for trsubset in sets:
			print 'random', count," ", len(trsubset)
			count += 1
			x, y = getftlbls(trsubset)
			#print 'kernel : ',trkernel
			grid = GridSearchCV(SVC(kernel = trkernel), param_grid=param_grid, cv=cv, n_jobs = 2)
			grid.fit(x, y)
			tempg.append(grid.best_params_['gamma'])
			tempc.append(grid.best_params_['C'])
			#print("For %s kernel best parameters are %s with a score of %0.2f" % (trkernel, grid.best_params_, grid.best_score_))
		
		print "data Percent: ",dpercent , "Kernel : ", trkernel, " avg gamma: ", sum(tempg)/len(tempg), " avg C: ", sum(tempc)/len(tempc)

		'''
		plt.plot(vacc, label = 'val')
		plt.legend(loc = 'upper right')
		plt.xlabel('k')
		plt.ylabel('Accuracy Val')
		pltitle = 'Acc vs Neighbours, ' + str(kf) + ' fold validation'
		plt.title(pltitle)
		plt.savefig('covdata percent : ' + str(dpercent) + ' folds' + str(kf) + '.png')
		plt.close()
		'''


'''
CovData
data Percent:  50 Kernel :  rbf  avg gamma:  1.19389019887e-07  avg C:  571646.306452
data Percent:  50 Kernel :  poly  avg gamma:  1e-05  avg C:  1e-05
data Percent:  50 Kernel :  sigmoid  avg gamma:  1e-09  avg C:  1e-09
data Percent:  50 Kernel :  linear  avg gamma:  1e-09  avg C:  7.71969016776e-06

data Percent:  75 Kernel :  poly  avg gamma:  1e-05  avg C:  0.0642314285714
data Percent:  75 Kernel :  sigmoid  avg gamma:  8.67395061676e-07  avg C:  0.0361434571429
data Percent:  75 Kernel :  rbf  avg gamma:  2.59045663587e-07  avg C:  0.0557147285714
data Percent:  75 Kernel :  linear  avg gamma:  1e-09  avg C:  0.0262470571429

data Percent:  100 Kernel :  poly  avg gamma:  1e-05  avg C:  0.0610774193548
data Percent:  100 Kernel :  sigmoid  avg gamma:  6.10351825817e-07  avg C:  0.0391403548387
data Percent:  100 Kernel :  rbf  avg gamma:  1e-09  avg C:  1e-06
data Percent:  100 Kernel :  linear  avg gamma:  1e-09  avg C:  0.00243446499852


MNIST
data Percent:  50 Kernel :  linear  avg gamma:  1e-09  avg C:  0.01
data Percent:  50 Kernel :  poly  avg gamma:  6.85e-06  avg C:  0.0415
data Percent:  50 Kernel :  rbf  avg gamma:  8.65e-07  avg C:  9.55
data Percent:  50 Kernel :  sigmoid  avg gamma:  6.4e-08  avg C:  86.05

data Percent:  75 Kernel :  linear  avg gamma:  1e-09  avg C:  0.01
data Percent:  75 Kernel :  poly  avg gamma:  7.6e-06  avg C:  0.064
data Percent:  75 Kernel :  rbf  avg gamma:  9.1e-07  avg C:  9.4
data Percent:  75 Kernel :  sigmoid  avg gamma:  5.44e-08  avg C:  174.4

data Percent:  100 Kernel :  linear  avg gamma:  1e-09  avg C:  0.01
data Percent:  100 Kernel :  poly  avg gamma:  7.94285714286e-06  avg C:  0.0562857142857
data Percent:  100 Kernel :  rbf  avg gamma:  9.22857142857e-07  avg C:  6.65714285714
data Percent:  100 Kernel :  sigmoid  avg gamma:  5.88571428571e-08  avg C:  102.314285714

pd = {}
pd[(50, 'rbf')] = [1.19389019887e-07, 571646.306452]
pd[(50, 'poly')] = [1e-05, 1e-05]
pd[(50, 'sigmoid')] = [1e-09, 1e-09]
pd[(50, 'linear')] = [1e-09, 7.71969016776e-06]

pd[(75, 'rbf')] = [2.59045663587e-07, 0.0557147285714]
pd[(75, 'poly')] = [1e-05, 0.0642314285714]
pd[(75, 'sigmoid')] = [8.67395061676e-07, 0.0361434571429]
pd[(75, 'linear')] = [1e-09, 0.0262470571429]

pd[(100, 'rbf')] = [1e-09, 1e-06]
pd[(100, 'poly')] = [1e-05, 0.0610774193548]
pd[(100, 'sigmoid')] = [6.10351825817e-07, 0.0391403548387]
pd[(100, 'linear')] = [1e-09, 0.00243446499852]

trkernel = 'rbf'
g = 1.19389019887e-07
c = 571646.306452
for dpercent in [1, 50]:
	dsub = subset(dpercent, traindata)
	y = [r[0] for r in dsub] 
	x = [r[1:] for r in dsub] 
	svc = SVC(kernel = trkernel, C = c, gamma = g)
	print dpercent, " ", trkernel, " ", svc.fit(x, y).score(xt, yt)

'''

