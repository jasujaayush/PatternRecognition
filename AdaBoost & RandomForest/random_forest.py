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
import decision_tree_new as dt

class Random_Forest:

	def __init__(self,dataset,n_trees,bootstrap_size,m_features, max_depth = sys.maxint):
		self.dataset = dataset
		self.rf_trees = []
		self.n_trees = n_trees
		self.bootstrap_size = bootstrap_size
		self.m_features = m_features
		self.max_depth = max_depth
		self.out_of_bag_tracker = defaultdict(list)
			
	def get_trees(self, n):
		if n > self.n_trees or self.n_trees == 0:
			print "People are so dumb! Forest can't return more than itself!"
			return []
		trees = []
		for i in range(n):
			trees.append(self.rf_trees[i])
		return trees	

	def create_trees(self):
		for i in range(0, self.n_trees):
			print "Making Tree - " + str(i)
			rf_tree = dt.Tree(self.bootstrap_size, self.m_features, self.max_depth)
			rows = rf_tree.setReducedData(self.dataset)
			for r in rows:
				self.out_of_bag_tracker[r].append(i)
			rf_tree.build_tree(self.max_depth)
			self.rf_trees.append(rf_tree)

	def predictDataset(self,predict_set, numtrees = -1):
		if numtrees != -1:
			trees = self.get_trees(numtrees)
		else:
			trees = self.rf_trees

		true = np.array([d[0] for d in predict_set])
		vote_pred = np.array([0.0 for d in predict_set])
		i = 0
		for d in predict_set:
			preds = np.array([tree.predict(d) for tree in trees])
			#if i%100 == 0:
			#	print i, preds, true[i]
			vote_pred[i] = Counter(preds).most_common(1)[0][0]
			i += 1
		acc = sum(true == vote_pred)/float(len(true))
		return acc, true, vote_pred
	
	def outofbagPredictDataset(self, dataset, numtrees = -1):
		if numtrees == -1:
			numtrees = self.n_trees

		true = []
		vote_pred = []		
		for i in range(len(dataset)):
			d = dataset[i]
			preds = np.array([self.rf_trees[j].predict(d) for j in range(0,numtrees) if j not in self.out_of_bag_tracker[i]])
			if len(preds) > 0:
				true.append(d[0])
				vote_pred.append(Counter(preds).most_common(1)[0][0])
		true = np.array(true)		
		vote_pred = np.array(vote_pred)
		acc = sum(true == vote_pred)/float(len(true))
		return acc, true, vote_pred

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
t = rf.parseData('covtype.data')
data = [[d[-1]] + d[:-1] for d in t]

import random_forest as rf
import numpy as np
data = rf.parseData('train.csv')
x = int(.75*len(data))
train = np.array(data[:x])
val = np.array(data[x:])
M = 50
ntrees = 80
bootstrap_size = 5000
max_depth = 5
rfc = rf.Random_Forest(train,ntrees,bootstrap_size,M,max_depth)
rfc.create_trees()
acc, true, pred = rfc.predictDataset(val)


t = rfc.rf_trees[0]
gs = t.giniscores
indi = t.columns
val = [gs[k] for k in gs]
import matplotlib.pyplot as plt
plt.plot(val, label = 'gini index')
plt.legend(loc = 'upper right')
plt.xlabel('features')
plt.ylabel('gini')
plt.title('Feature Gini Index')
plt.show()

tr, te, ob = [], [] ,[]
for n in range(1,ntrees+1, 5):
	print n
	x,y,z = rfc.predictDataset(train,n)
	tr.append(100 - x*100)
	x,y,z = rfc.outofbagPredictDataset(train, n)
	ob.append(100 - x*100)
	x,y,z = rfc.predictDataset(val,n)
	te.append(100 - x*100)

import matplotlib.pyplot as plt
plt.plot(te, label = 'test')
plt.plot(tr, label = 'train')
plt.plot(ob, label = 'oob')
plt.legend(loc = 'upper right')
plt.xlabel('Number of Trees / 5')
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

from sklearn.ensemble import RandomForestClassifier
rfcl = RandomForestClassifier(n_estimators=ntrees, bootstrap=True, max_depth=5)
y = [r[0] for r in train]
x = [r[1:] for r in train]
rfcl.fit(x,y)
print rfcl.score(x,y)
yv = [r[0] for r in val]
xv = [r[1:] for r in val]
rfcl.score(xv,yv)

'''