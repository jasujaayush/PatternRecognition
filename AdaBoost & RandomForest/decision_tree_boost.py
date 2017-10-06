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

class Tree:	
	def __init__(self, max_depth = sys.maxint):
		self.root = None
		self.max_depth = max_depth
		self.alpha = 1

	def get_root(self):
		return self.root

	def get_alpha(self):
		return self.alpha

	def randomSplit(self,index,value,current_set, weights):
		left, right = list(), list()
		lweight, rweight = list(), list()
		for i in range(len(current_set)):
			row = current_set[i]
			w = weights[i]
			if row[index] < value:
				left.append(row)
				lweight.append(w)
			else:
				right.append(row)
				rweight.append(w)
		return (left,lweight),(right,rweight)

	def giniIndex(self,groups,classes):
		gini = 0.0
		total = 0
		for group in groups:
			total += sum(group[1])

		for group in groups:
			value = 0
			size = len(group[1])
			if size == 0:
				continue
			temp = [row[0] for row in group[0]]		
			for class_value in classes:	
				proportion = temp.count(class_value) / float(size)
				value += (proportion * proportion)
			gini +=	(sum(group[1])/float(total))*value
		return gini	

	def getSplit(self,current_set, data_weight):
		classes = list(set(row[0] for row in current_set))
		b_index, b_value, b_score, b_groups = -999999, -999999, -999999, None
		M = len(current_set[0])
		weights = [data_weight[i] for i in range(len(current_set))]
		for index in range(1, M):
			#print index
			#np.unique(current_set[:,index])
			for val in range(0,256,25):
				groups = self.randomSplit(index, val, current_set, weights)
				gini = self.giniIndex(groups, classes)
				if gini > b_score:
					b_index, b_value, b_score, b_groups = index, val, gini, (groups[0][0],groups[1][0])

		#print('best split: ' + 'X%d > %.3f Gini=%.3f' % ((b_index), row[b_index], b_score))
		return {'index':b_index, 'value':b_value, 'groups':b_groups}

	def make_terminal(self,group):
		classes = [data[0] for data in group]
		mode = Counter(classes).most_common(1)
		return mode[0][0]

	def build_tree_recurse(self,node,max_depth, min_bucket_size,current_depth, data_weight):
		left_set, right_set = node['groups']
		del(node['groups'])
		if left_set == [] :
			node['left'] = node['right'] = self.make_terminal(right_set)
			return
		if right_set == []:
			node['left'] = node['right'] = self.make_terminal(left_set)
			return
		if current_depth > max_depth:
			node['left'] = self.make_terminal(left_set)
			node['right'] = self.make_terminal(right_set)
			print 'Max Depth Reached! ', max_depth, current_depth
			return
		if len(left_set) <= min_bucket_size:
			node['left'] = self.make_terminal(left_set)
		else:
			node['left'] = self.getSplit(left_set, data_weight)
			self.build_tree_recurse(node['left'],max_depth,min_bucket_size,current_depth+1, data_weight)
		if len(right_set) <= min_bucket_size:
			node['right'] = self.make_terminal(right_set)
		else:
			node['right'] = self.getSplit(right_set, data_weight)
			self.build_tree_recurse(node['right'],max_depth,min_bucket_size,current_depth+1, data_weight)

	def build_tree(self, dataset, data_weight, max_depth=sys.maxint,min_bucket_size=1):
		self.root = self.getSplit(dataset, data_weight)
		self.build_tree_recurse(self.root,max_depth,min_bucket_size,1, data_weight)
		data_weight = self.reweight(dataset, data_weight)	
		return data_weight

	def predict(self,data):
		return self.predict_recurse(self.root,data)

	def predict_recurse(self,node,data):
		split_feature = node['index']
		if data[split_feature] < node['value']:
			if isinstance(node['left'], dict):
				return self.predict_recurse(node['left'], data)
			else:
				return node['left']
		else:
			if isinstance(node['right'], dict):
				return self.predict_recurse(node['right'], data)
			else:
				return node['right']

	def reweight(self, dataset, data_weight):
		t = [d[0] for d in dataset]
		nclasses = len(set(t))
		r = [self.predict(d) for d in dataset]
		err = 0.0
		total = 0.0
		for i in range(len(t)):
			total += data_weight[i]
			if t[i] != r[i]:
				err += data_weight[i]
		err = err/float(total)
		alpha = log((1-err)/err) + log(nclasses - 1)

		total = 0
		for i in range(len(t)):
			if t[i] != r[i]:
				data_weight[i] = data_weight[i]*exp(alpha)
			total += data_weight[i]
		for k in data_weight:
			data_weight[k] = data_weight[k]/total

		self.alpha = alpha	
		return data_weight
