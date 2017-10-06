import numpy as np
from math import exp
from math import log
import scipy.optimize
from random import shuffle
import seaborn as sns                                                  
import matplotlib as mpl                                               
import matplotlib.pyplot as plt                                        
import pandas as pd
import numpy

class NB:
	def getParams(self,data, n = 5):
		features1 = []
		features2 = []
		for x in data:
			if int(x[0]) == 1:
				feat = []
				for v in x[1:]:
					feat.append(float(v))
				features1.append(feat)
			if int(x[0]) == 2:
				feat = []
				for v in x[1:]:
					feat.append(float(v))
				features2.append(feat)
		
		features = features1 + features2
		xtrain = numpy.array(features1[:n] + features2[:n])
		mean = xtrain.mean(0)
		mean1 = numpy.array(features1).mean(0)
		mean2 = numpy.array(features2).mean(0)

		d = len(features[0])
		scatter_matrix = np.zeros((d,d))
		for i in range(len(xtrain)):
			sample = np.array(xtrain[i])
			scatter_matrix += (sample - mean).T.dot((sample - mean))
		cov_mat = scatter_matrix
		#sigma = cov_mat.sum(0)
		sigma = numpy.var(xtrain, axis=0)

		w = []
		pie = len(features1)*1.0/len(features)
		w0 = log((1-pie)/pie)
		for i in range(len(xtrain[0])):
			w0 += (mean2[i]**2 - mean1[i]**2)/(2*sigma[i])
		w.append(w0)	
		for i in range(len(xtrain[0])):
			t = (mean1[i] - mean2[i])/(sigma[i])
			w.append(t)

		return w	
