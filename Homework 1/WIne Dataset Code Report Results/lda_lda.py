import numpy as np
import urllib
import scipy.optimize
import random
from math import exp
from math import log
import csv
from random import shuffle
import scipy.linalg as linalg
import numpy
import lda_compressor
import pca_compressor
import matplotlib.pyplot as plt
import pandas as pd

#Compress
n = 6
lda = lda_compressor.LDA()
data = lda.parseData('wine.csv')
shuffle(data)
features1,label1,features2,label2 = lda.getFeatures(data)
eig_pairs, mean = lda.getEigenValues(features1, label1, features2, label2, n)

#Take 1 eigen vector and transform
features = features1[n:] + features2[n:]
label = label1[n:] + label2[n:]
data, matrix_w = lda.compressor(eig_pairs, features, label, 1)

#Classify
lda.classifier(data)

#Plot Eigen Values
s = 0
t = []
for e in eig_pairs:
    s += e[0]
    t.append(s)

for i in range(len(t)):
    t[i] /= s

x = range(len(eig_pairs))
plt.xlabel('eigen value index')
plt.ylabel('cumulative eigen value/total')
plt.plot(x, t)
plt.show()

#Plot Data
pca = pca_compressor.PCA()
features = features1[:n] + features2[:n]
label = label1[:n] + label2[:n]
data, matrix_w = lda.compressor(eig_pairs, features, label, 1)
x = features
y = label
wine_df3_pca = pd.DataFrame(x)
wine_df3_pca['Class'] = y
axarr,c = pca.factor_scatter_matrix(wine_df3_pca,'Class')
plt.show(axarr)

#Reconstruction Error
f1 = numpy.matrix(features[15])
print f1
d = (f1 - mean).dot(matrix_w)
print d
wi = linalg.pinv(matrix_w)
f2 = d.dot(matrix_w.T) + mean
print f2
e = f2-f1
e1 = e.dot(e.T)
print e1
'''
orig: [[  1.31100000e+01   1.01000000e+00   1.70000000e+00   1.50000000e+01
    7.80000000e+01   2.98000000e+00   3.18000000e+00   2.60000000e-01
    2.28000000e+00   5.30000000e+00   1.12000000e+00   3.18000000e+00
    5.02000000e+02]]
trans : [[-38.44693374]]
recons : [[  1.26098663e+01   1.16375673e+00   1.92981549e+00   1.61887738e+01
    9.29335214e+01   2.45559062e+00   1.74116629e+00   3.58886738e-01
    9.35610936e-01   4.06392481e+00   1.12318970e+00   2.13423570e+00
    5.01569461e+02]]
error : [[ 231.71912201]]
'''


X = []
y = []
i = 0
j = 0
for d in data:
    if d[0] == 1 and i < 5:
        i += 1
        X.append(d[1:])
        y.append(d[0])
    elif d[0] == 2 and j < 5:
        j += 1
        X.append(d[1:])
        y.append(d[0])
    elif j>= 5 and i >= 5:
        break

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# LDA
sklearn_lda = LDA(n_components=5)
X_lda_sklearn = sklearn_lda.fit_transform(X, y)



#Plot
pca = pca_compressor.PCA()
x = [d[1:] for d in data]
y = [d[0] for d in data]
wine_df3_pca = pd.DataFrame(x)
wine_df3_pca['Class'] = y
axarr,c = pca.factor_scatter_matrix(wine_df3_pca,'Class')
plt.show(axarr)