import numpy as np
import urllib
import scipy.optimize
import random
from math import exp
from math import log
import csv
from random import shuffle

def parseData(fname):
  with open(fname, 'rU') as f:
    reader = csv.reader(f)
    data = list(list(rec) for rec in csv.reader(f, delimiter=','))
  return data       

data = parseData("wine.csv")
shuffle(data)

features1 = []
label1 = []
features2 = []
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

n = 5
xtrain1 = [[1.0] + f for f in features1[:n]]
ytrain1 = [y for y in label1[:n]]
xtrain2 = [[1.0] + f for f in features2[:n]]
ytrain2 = [y for y in label2[:n]]

mtrain1 = np.matrix(xtrain1)
mean1 = mtrain1.mean(0)
mtrain2 = np.matrix(xtrain2)
mean2 = mtrain2.mean(0)

d = len(xtrain1[0])
cov_mat1 = np.zeros((d,d))
for i in range(len(xtrain1)):
    sample = np.array(xtrain1[i])
    cov_mat1 += (sample - mean1).T.dot((sample - mean1))
cov_mat1 = cov_mat1/n    
print('Covariance Matrix 1:\n', cov_mat1)

d = len(xtrain2[0])
cov_mat2 = np.zeros((d,d))
for i in range(len(xtrain2)):
    sample = np.array(xtrain2[i])
    cov_mat2 += (sample - mean2).T.dot((sample - mean2))
cov_mat2 = cov_mat2/n    
print('Covariance Matrix 2:\n', cov_mat2)

cov_mat = cov_mat1 + cov_mat2
inv_cov = linalg.pinv(cov_mat)
mdiff = mean1 - mean2
weight = inv_cov.dot(mdiff.T)
weight = weight.T
mavg = (mean1 + mean2)/2.0
threshold = weight.dot(mavg.T)
threshold = float(threshold)

features = features1[n:] + features2[n:]
xtest = [[1.0] + f for f in features]
label = label1[n:] + label2[n:]
ytest = [y for y in label]

results = []
for x in xtest:
    x = numpy.matrix(x)
    val = weight.dot(x.T)
    val = float(val)
    if val > threshold:
        results.append(1)
    else:
        results.append(2) 

a = numpy.array(results)           
b = numpy.array(ytest)
c = (a == b)