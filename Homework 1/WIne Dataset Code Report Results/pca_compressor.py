import csv
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

class PCA:
	def parseData(self, fname):
		with open(fname, 'rU') as f:
			reader = csv.reader(f)
			data = list(list(rec) for rec in csv.reader(f, delimiter=','))
		return data      
	
	def getFeatures(self, data):
		features = []
		features1 = []
		features2 = []
		label = []
		label1 = []
		label2 = []
		for x in data:
			if int(x[0]) == 1:
				label1.append(int(x[0]))
				feat = []
				for v in x[1:]:
					feat.append(float(v))
				features1.append(feat)
			if int(x[0]) == 2:
				label2.append(int(x[0]))
				feat = []
				for v in x[1:]:
					feat.append(float(v))
				features2.append(feat)
		return features1,label1,features2, label2
				
	def getEigenValues(self, features1, label1, features2, label2, n = 5):
		features = features1[:n] + features2[:n]
		label = label1[:n] + label2[:n]
		xtrain = [f for f in features]
		ytrain = [y for y in label]

		mtrain = np.matrix(xtrain)
		mean = mtrain.mean(0)

		d = len(xtrain[0])
		scatter_matrix = np.zeros((d,d))
		for i in range(len(xtrain)):
			sample = np.array(xtrain[i])
			scatter_matrix += (sample - mean).T.dot((sample - mean))
		print('Scatter Matrix:\n', scatter_matrix)

		eig_val_sc, eig_vec_sc = np.linalg.eigh(scatter_matrix)

		eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]
		eig_pairs.sort(key=lambda x: x[0], reverse=True)
		for i in eig_pairs:
			print(i[0])

		return eig_pairs, mean
			
	def compressor(self, eig_pairs, features, label, deigen = 5):
		matrix_w = []
		for i in range(deigen):
			matrix_w.append(eig_pairs[i][1])
		matrix_w = np.matrix(matrix_w)
		matrix_w = matrix_w.T
		#print('Matrix W:\n', matrix_w)
		
		x = np.matrix(features)
		tx = x.dot(matrix_w)

		features = tx.tolist()
		data = [[label[i]] + features[i] for i in range(len(features))]
		return data, matrix_w

	# This code is taken from Stack_Overflow for getting a scatter matrix.
	def factor_scatter_matrix(self, df, factor, palette=None):
		'''
		Create a scatter matrix of the variables in df, with differently colored
		points depending on the value of df[factor].
		inputs:
		    df: pandas.DataFrame containing the columns to be plotted, as well 
		        as factor.
		    factor: string or pandas.Series. The column indicating which group 
		        each row belongs to.
		    palette: A list of hex codes, at least as long as the number of groups.
		        If omitted, a predefined palette will be used, but it only includes
		        9 groups.
		'''
		import matplotlib.colors
		import numpy as np
		from pandas.tools.plotting import scatter_matrix
		from scipy.stats import gaussian_kde

		if isinstance(factor, basestring):
		    factor_name = factor #save off the name
		    factor = df[factor] #extract column
		    df = df.drop(factor_name,axis=1) # remove from df, so it 
		    # doesn't get a row and col in the plot.

		classes = list(set(factor))

		if palette is None:
		    palette = ['#e41a1c', '#377eb8', '#4eae4b', 
		               '#994fa1', '#ff8101', '#fdfc33', 
		               '#a8572c', '#f482be', '#999999']

		color_map = dict(zip(classes,palette))

		if len(classes) > len(palette):
		    raise ValueError('''Too many groups for the number of colors provided.
		We only have {} colors in the palette, but you have {}
		groups.'''.format(len(palette), len(classes)))

		colors = factor.apply(lambda group: color_map[group])
		axarr = scatter_matrix(df,figsize=(10,10),marker='o',c=colors,diagonal=None)


		for rc in xrange(len(df.columns)):
		    for group in classes:
		        y = df[factor == group].icol(rc).values
		        gkde = gaussian_kde(y)
		        ind = np.linspace(y.min(), y.max(), 1000)
		        axarr[rc][rc].plot(ind, gkde.evaluate(ind),c=color_map[group])

		return axarr, color_map