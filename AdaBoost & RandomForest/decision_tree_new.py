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
	def __init__(self, bootstrap_size, m_features, max_depth = sys.maxint):
		self.dataset = None
		self.root = None
		self.bootstrap_size = bootstrap_size
		self.m_features = m_features
		self.columns = []
		self.giniscores = defaultdict(float)
		self.max_depth = max_depth

	def get_root(self):
		return self.root

	def setReducedData(self,dataset):
		l = len(dataset[0]) - 1
		columns = [random.randint(1,l) for i in range(self.m_features)]
		#columns = range(1,self.m_features)
		rows = [random.randint(0,len(dataset) - 1) for i in range(self.bootstrap_size)]
		columns = [0] + columns
		rdata = dataset[np.ix_(rows,columns)]
		self.columns = columns
		self.dataset = rdata
		return rows	

	def randomSplit(self,index,value,current_set):
		left, right = list(), list()
		for row in current_set:
			if row[index] < value:
				left.append(row)
			else:
				right.append(row)
		return left, right

	def giniIndex(self,groups,classes):
		gini = 0.0
		total = 0
		for group in groups:
			total += len(group)

		for group in groups:
			value = 0
			size = len(group)
			if size == 0:
				continue
			temp = [row[0] for row in group]		
			for class_value in classes:	
				proportion = temp.count(class_value) / float(size)
				value += (proportion * proportion)
			gini +=	(len(group)/float(total))*value
		return gini

	def getSplit(self,current_set):
		classes = list(set(row[0] for row in current_set))
		b_index, b_value, b_score, b_groups = -999999, -999999, -999999, None
		M = len(current_set[0])
		for index in range(1, M):
			self.giniscores[index] = -999999
			for val in range(0,257):
				groups = self.randomSplit(index, val, current_set)
				gini = self.giniIndex(groups, classes)
				if gini > self.giniscores[index]:
					self.giniscores[index] = gini
				if gini > b_score:
					#l = [d[0] for d in groups[0]]						
					#r = [d[0] for d in groups[1]]						
					b_index, b_value, b_score, b_groups = index, val, gini, groups

		#print('best split: ' + 'X%d > %.3f Gini=%.3f' % ((b_index), row[b_index], b_score))
		return {'index':b_index, 'value':b_value, 'groups':b_groups}

	def make_terminal(self,group):
		classes = [data[0] for data in group]
		mode = Counter(classes).most_common(1)
		return mode[0][0]

	def build_tree_recurse(self,node,max_depth, min_bucket_size,current_depth):
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
			node['left'] = self.getSplit(left_set)
			self.build_tree_recurse(node['left'],max_depth,min_bucket_size,current_depth+1)
		if len(right_set) <= min_bucket_size:
			node['right'] = self.make_terminal(right_set)
		else:
			node['right'] = self.getSplit(right_set)
			self.build_tree_recurse(node['right'],max_depth,min_bucket_size,current_depth+1)

	def build_tree(self,max_depth=sys.maxint,min_bucket_size=1):
		self.root = self.getSplit(self.dataset)
		self.build_tree_recurse(self.root,max_depth,min_bucket_size,1)

	def predict(self,data):
		rd = data[self.columns]
		return self.predict_recurse(self.root,rd)


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