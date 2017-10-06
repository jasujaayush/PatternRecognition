import csv
import sys
import urllib
import random
import numpy as np
import pandas as pd
from math import exp
from math import log
import scipy.optimize
from random import shuffle
from sklearn.svm import SVC
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
'''
temp1 = parseData('covtype.data')
t = int(.15*len(temp1))  #take 15 percent of train data
temp1 = temp1[:t]
temp = [[d[-1]] + d[:-1] for d in temp1]
del temp1
'''
temp = parseData('train.csv')  
x = int(.75*len(temp))
testdata = temp[x:]
traindata = temp[:x]

pca = PCA(n_components=30)
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
testdata = [[y[i]] + xt[i].tolist() for i in range(len(xt))]

pd = {}

pd[(50, 'rbf')] = [8.65e-07, 9.55]
pd[(50, 'poly')] = [6.85e-06, 0.0415]
pd[(50, 'sigmoid')] = [6.4e-08, 86.05]
pd[(50, 'linear')] = [1e-09, 0.01]

pd[(75, 'rbf')] = [9.1e-07, 9.4]
pd[(75, 'poly')] = [7.6e-06, 0.064]
pd[(75, 'sigmoid')] = [5.44e-08, 174.4]
pd[(75, 'linear')] = [1e-09,  0.01]

pd[(100, 'rbf')] = [9.22857142857e-07, 6.65714285714]
pd[(100, 'poly')] = [7.94285714286e-06, 0.0562857142857]
pd[(100, 'sigmoid')] = [5.88571428571e-08, 102.314285714]
pd[(100, 'linear')] = [1e-09, 0.01]

for trkernel in ['rbf', 'poly', 'sigmoid','linear']:
        print 'processing ', trkernel
        for dpercent in [50, 75, 100]:
                dsub = subset(dpercent, traindata)
                y = [r[0] for r in dsub]
                x = [r[1:] for r in dsub]
                g = pd[(dpercent, trkernel)][0]
                c = pd[(dpercent, trkernel)][1]
                svc = SVC(kernel = trkernel, C = c, gamma = g)
                #print dpercent, " ", trkernel, " ", svc.fit(x, y).score(xt, yt)
                scores = cross_val_score(svc, x, y, cv=5)
                print dpercent, " ", trkernel, " ", sum(scores)/len(scores)
'''
MNIST
pd[(50, 'rbf')] = [8.65e-07, 9.55]
pd[(50, 'poly')] = [6.85e-06, 0.0415]
pd[(50, 'sigmoid')] = [6.4e-08, 86.05]
pd[(50, 'linear')] = [1e-09, 0.01]

data Percent:  50 Kernel :  linear  avg gamma:  1e-09  avg C:  0.01
data Percent:  50 Kernel :  poly  avg gamma:  6.85e-06  avg C:  0.0415
data Percent:  50 Kernel :  rbf  avg gamma:  8.65e-07  avg C:  9.55
data Percent:  50 Kernel :  sigmoid  avg gamma:  6.4e-08  avg C:  86.05

pd[(75, 'rbf')] = [9.1e-07, 9.4]
pd[(75, 'poly')] = [7.6e-06, 0.064]
pd[(75, 'sigmoid')] = [5.44e-08, 174.4]
pd[(75, 'linear')] = [1e-09,  0.01]

data Percent:  75 Kernel :  linear  avg gamma:  1e-09  avg C:  0.01
data Percent:  75 Kernel :  poly  avg gamma:  7.6e-06  avg C:  0.064
data Percent:  75 Kernel :  rbf  avg gamma:  9.1e-07  avg C:  9.4
data Percent:  75 Kernel :  sigmoid  avg gamma:  5.44e-08  avg C:  174.4

pd[(100, 'rbf')] = [9.22857142857e-07, 6.65714285714]
pd[(100, 'poly')] = [7.94285714286e-06, 0.0562857142857]
pd[(100, 'sigmoid')] = [5.88571428571e-08, 102.314285714]
pd[(100, 'linear')] = [1e-09, 0.01]

data Percent:  100 Kernel :  linear  avg gamma:  1e-09  avg C:  0.01
data Percent:  100 Kernel :  poly  avg gamma:  7.94285714286e-06  avg C:  0.0562857142857
data Percent:  100 Kernel :  rbf  avg gamma:  9.22857142857e-07  avg C:  6.65714285714
data Percent:  100 Kernel :  sigmoid  avg gamma:  5.88571428571e-08  avg C:  102.314285714
'''