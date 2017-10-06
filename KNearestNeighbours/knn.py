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
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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

def crossValSplit(f, data):
	t = int(len(data)/float(f))
	sets = []
	for i in range(f):
		temp = data[i*t : t*(i+1)]
		sets.append(temp)
	return sets

temp1 = parseData('covtype.data')	
temp = [[d[-1]] + d[:-1] for d in temp1[:400000]]
del temp1
#temp = parseData('train.csv')	
x = int(.75*len(temp))
testdata = temp[x:]
traindata = temp[:x]


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


for dpercent in [50, 75, 100]:
	trsubset = subset(dpercent, traindata)
	for kf in [2, 5, len(trsubset)]:
		print 'processing : '+str(dpercent), " folds: ", kf
		sets = crossValSplit(kf, trsubset)
		standard_scaler = StandardScaler()
		maxk = 200
		krange = range(1,7)
		l = len(krange)
		vacc = np.zeros(l)
		if kf == 5 or kf == 2:
			for i in range(len(sets)):
				val = sets[i]
				trainsets = sets[:i] + sets[i+1:]
				train = [d for s in trainsets for d in s ]
				y = [r[0] for r in train]
				x = standard_scaler.fit_transform([r[1:] for r in train])
				yv = [r[0] for r in val]
				xv = standard_scaler.transform([r[1:] for r in val])
				for k in krange:
					vacc[k-1] += KNeighborsClassifier(n_neighbors= k, algorithm='ball_tree').fit(x,y).score(xv, yv)
					print 'kfold',kf,' Set',i,' - Done with ',k
			vacc = vacc/len(sets)		
		else:
			for i in range(1,len(sets),100): #cross validating with kf/100 sets
				train = trsubset[:i] + trsubset[i+1:]
				val = [trsubset[i]]
				y = [r[0] for r in train]
				x = standard_scaler.fit_transform([r[1:] for r in train])
				yv = [r[0] for r in val]
				xv = standard_scaler.transform([r[1:] for r in val])
				for k in krange:
					vacc[k-1] += KNeighborsClassifier(n_neighbors= k, algorithm='ball_tree').fit(x,y).score(xv, yv)
					if k%2 == 0:
						print 'kfold',kf,' Set',i,' - Done with ',k				
			vacc = vacc/len(range(1,len(sets),100))

		
		plt.plot(vacc, label = 'val')
		plt.legend(loc = 'upper right')
		plt.xlabel('k')
		plt.ylabel('Accuracy Val')
		pltitle = 'Acc vs Neighbours, ' + str(kf) + ' fold validation'
		plt.title(pltitle)
		plt.savefig('covdata percent : ' + str(dpercent) + ' folds' + str(kf) + '.png')
		plt.close()
'''
yt = [r[0] for r in testdata] 
xt = [r[1:] for r in testdata]
standard_scaler = StandardScaler()
for dpercent in [1, 50, 75, 100]:
	trsubset = subset(dpercent, traindata)
	y = [r[0] for r in trsubset]
	x = standard_scaler.fit_transform([r[1:] for r in trsubset])
	xt = standard_scaler.transform(xt)
	if dpercent < 100:
		for kf in [1]:
			print dpercent, " ", kf, " ", KNeighborsClassifier(n_neighbors= kf, algorithm='ball_tree').fit(x,y).score(xt, yt)
	else:
		for kf in [1, 5]:
			print dpercent, " ", kf, " ", KNeighborsClassifier(n_neighbors= kf, algorithm='ball_tree').fit(x,y).score(xt, yt)		
'''