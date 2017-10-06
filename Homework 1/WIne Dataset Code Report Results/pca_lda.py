import pca_compressor
import lda_compressor
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import scipy.linalg as linalg                                      
from sklearn.decomposition import PCA
import nb
from random import shuffle

#Compress
n = 10
pca = pca_compressor.PCA()
data = pca.parseData('wine.csv')
shuffle(data)
features1,label1,features2,label2 = pca.getFeatures(data)
eig_pairs, mean = pca.getEigenValues(features1, label1, features2, label2, n)

#Take 1 eigen vector and transform
features = features1[n:] + features2[n:]
label = label1[n:] + label2[n:]
data, matrix_w = pca.compressor(eig_pairs, features, label, 13)

#Classify
lda = lda_compressor.LDA()
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
features = features1[:n] + features2[:n]
label = label1[:n] + label2[:n]
data, matrix_w = pca.compressor(eig_pairs, features, label, 1)
x = features
y = label
wine_df3_pca = pd.DataFrame(x)
wine_df3_pca['Class'] = y
axarr,c = pca.factor_scatter_matrix(wine_df3_pca,'Class')
plt.show(axarr)

#Reconstruction Error
f1 = numpy.matrix(features[15])
print f1
d = (f1-mean).dot(matrix_w)
print d
wi = linalg.pinv(matrix_w)
f2 = d.dot(matrix_w.T) + mean
print f2
e = f2-f1
e1 = e.dot(e.T)
print e1
'''
componenets = 1
original : [[  1.36300000e+01   1.81000000e+00   2.70000000e+00   1.72000000e+01
    1.12000000e+02   2.85000000e+00   2.91000000e+00   3.00000000e-01
    1.46000000e+00   7.30000000e+00   1.28000000e+00   2.88000000e+00
    1.31000000e+03]]
reduced dimension : [[ 520.05333809]]
reconstructed : [[  1.39353795e+01   1.82850552e+00   2.47380709e+00   1.66436662e+01
    1.09602176e+02   2.92995133e+00   3.08394307e+00   2.77114236e-01
    1.97129468e+00   6.05404282e+00   1.08792535e+00   3.11503645e+00
    1.31004585e+03]]
Error : [[ 8.14907023]]

components = 5
[[  1.36300000e+01   1.81000000e+00   2.70000000e+00   1.72000000e+01
    1.12000000e+02   2.85000000e+00   2.91000000e+00   3.00000000e-01
    1.46000000e+00   7.30000000e+00   1.28000000e+00   2.88000000e+00
    1.31000000e+03]]
[[  5.20053338e+02  -2.40581674e+00   4.47151080e-01   7.52854575e-01
    5.83488902e-01]]
[[  1.40277852e+01   1.40746180e+00   2.53063475e+00   1.71972124e+01
    1.11996775e+02   3.08632834e+00   3.36889675e+00   2.76942929e-01
    2.00284850e+00   6.79979018e+00   1.10804618e+00   3.13202859e+00
    1.31000001e+03]]
Error : 
[[ 1.25392263]]
'''

#Library Reconstruction Error
pcal = PCA(n_components=1)
pcal.fit(features1[:n] + features2[:n])
f1 = features[15]
d = pcal.transform(f1)
f2 = pcal.inverse_transform(d)
e = f2-f1
e2 = e.dot(e.T)
print f1
print d
print f2
print e2