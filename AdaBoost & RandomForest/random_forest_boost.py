import csv
import sys
import urllib
import random
import numpy as np
from math import exp
from math import log
import scipy.optimize
from random import shuffle
import seaborn as sns                                                  
import matplotlib as mpl                                               
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
import decision_tree_boost as dt


class Random_Forest:

	def __init__(self,dataset,n_trees, max_depth = sys.maxint):
		self.dataset = dataset
		self.rf_trees = []
		self.n_trees = n_trees
		self.max_depth = max_depth
		self.data_weight = self.init_data_weight(dataset)
		self.alpha = []

	def init_data_weight(self, dataset):	
		data_weight = np.ones(len(dataset)) / len(dataset)
		return data_weight

	def get_trees(self, n):
		if n > self.n_trees or self.n_trees == 0:
			print "People are so dumb! Forest can't return more than itself!"
			return []
		trees = []
		for i in range(n):
			trees.append(self.rf_trees[i])
		return trees	

	def updateWeight(self, tree, xtrain, ytrain, data_weight):
		nclasses = len(set(ytrain))
		pred = tree.predict(xtrain)
		err = 0.0
		total = 0.0
		for i in range(len(ytrain)):
			total += data_weight[i]
			if ytrain[i] != pred[i]:
				err += data_weight[i]
		err = err/float(total)
		alpha = log((1-err)/err) + log(nclasses - 1)

		total = 0
		for i in range(len(ytrain)):
			if ytrain[i] != pred[i]:
				data_weight[i] = data_weight[i]*exp(alpha)
			total += data_weight[i]
		for k in range(len(data_weight)):
			data_weight[k] = data_weight[k]/total

		self.alpha.append(alpha)
		return data_weight

	def create_trees(self):	
		for i in range(0, self.n_trees):
			print "Making Tree - " + str(i)
			rf_tree = DecisionTreeClassifier(max_depth = self.max_depth)
			train = self.dataset
			yt = [r[0] for r in train]
			xt = [r[1:] for r in train]
			rf_tree.fit(xt, yt, sample_weight = self.data_weight)
			self.data_weight = self.updateWeight(rf_tree, xt, yt, self.data_weight)
			self.rf_trees.append(rf_tree)

	def predictDataset(self,dataset, numtrees = -1):
		if numtrees != -1:
			trees = self.get_trees(numtrees)
		else:
			trees = self.rf_trees
			numtrees = self.n_trees

		true = np.array([r[0] for r in dataset])
		x = np.array([r[1:] for r in dataset])
		classes = set(true)

		# For one tree get all the predictions
		treePreds = []
		for t in trees:
			treePreds.append(t.predict(x))

		# predictions for a point across all trees	
		preds = []
		for i in range(len(x)):
			temp = []
			for pred in treePreds:
				temp.append(pred[i])
			preds.append(temp)
		
		prediction = np.array([0.0 for d in range(len(x))])
		for d in range(len(x)):
		    max_val = -sys.maxint
		    argmax_k = None
		    for k in classes:
		        classifier_list = [i for i in range(numtrees) if preds[d][i]==k]
		        val = sum([self.alpha[i] for i in classifier_list])
		        if val>max_val:
		            max_val = val
		            argmax_k = k
		    prediction[d] = argmax_k               
		acc = sum(true==prediction)/(len(prediction)*1.0)
		return acc, true, prediction

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

'''
import random_forest_boost as rfb
import numpy as np
data = rfb.parseData('train.csv')
x = int(.75*len(data))
train = np.array(data[:x])
val = np.array(data[x:])
ntrees = 100
max_depth = 10
rfc = rfb.Random_Forest(train,ntrees, max_depth)
rfc.create_trees()
a,t,p = rfc.predictDataset(val)

tr, te = [], []
for n in range(1,ntrees+1):
	x,y,z = rfc.predictDataset(train,n)
	tr.append(100 - x*100)
	x,y,z = rfc.predictDataset(val,n)
	te.append(100 - x*100)

import matplotlib.pyplot as plt
plt.plot(te, label = 'test')
plt.plot(tr, label = 'train')
plt.legend(loc = 'upper right')
plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.title('Error vs Trees')
plt.show()

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
acc, true, pred = rfc.predictDataset(val)
cnf_matrix = confusion_matrix(true, pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
class_names = list(set(true))
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')    
plt.show()

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import random_forest_boost as rfb
import numpy as np
t = rfb.parseData('covtype.data')
data = [[d[-1]] + d[:-1] for d in t]
x = int(.75*len(data))
train = np.array(data[:x])
val = np.array(data[x:])
te = []
tr = []
for ntrees in range(1,100,5):
	print ntrees
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators= ntrees, algorithm='SAMME')
	yt = [r[0] for r in train]
	xt = [r[1:] for r in train]
	bdt.fit(xt,yt)
	a = bdt.score(xt,yt)
	tr.append(a)
	y = [r[0] for r in val]
	x = [r[1:] for r in val]
	a = bdt.score(x,y)
	te.append(a)
'''
