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
from scipy.stats import mode

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
t = int(.15*len(temp1))  #take 15 percent of train data
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
testdata = [[y[i]] + xt[i].tolist() for i in range(len(xt))]
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

for trkernel in ['linear'] :#['rbf', 'poly', 'sigmoid']:
        print 'processing ', trkernel
        for dpercent in [75, 100]: #50
                dsub = subset(dpercent, traindata)
                numsets = len(dsub)/700
                print numsets
                sets = fsplit(numsets, dsub)
                g = pd[(dpercent, trkernel)][0]
                c = pd[(dpercent, trkernel)][1]
                results = []
                crossval = []
                count = 0
                for trsubset in sets:
                        if count%5 == 0:
                                print 'set ', count
                        count += 1
                        x, y = getftlbls(trsubset)
                        svc = SVC(kernel = trkernel, C = c, gamma = g)
                        scores = cross_val_score(svc, x, y, cv=2)
                        crossval.append(sum(scores)/len(scores))
                        #svc.fit(x, y)
                        #results.append(svc.predict(xt))
                print dpercent, " ", trkernel, " ", sum(crossval)/float(len(crossval))        
                #yp = np.array(mode(results)[0])
                #yt = np.array(yt)
                #print yp, yt
                #print dpercent, " ", trkernel, " ", sum(yp[0] == yt)/float(len(yt))
