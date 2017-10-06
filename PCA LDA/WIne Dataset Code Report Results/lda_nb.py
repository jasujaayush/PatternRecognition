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
import nb
from random import shuffle
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import pca_compressor
import pandas as pd


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

n = 5
lda = lda_compressor.LDA()
data = lda.parseData('wine.csv')
shuffle(data)
features1,label1,features2,label2 = lda.getFeatures(data)
eig_pairs, mean = lda.getEigenValues(features1, label1, features2, label2, n)

#Take 1 eigen vector and transform
features = features1[n:] + features2[n:]
label = label1[n:] + label2[n:]
data, matrix_w = lda.compressor(eig_pairs, features, label, 1)

nb = nb.NB()
shuffle(data)
w =nb.getParams(data)
test = data
xtest = [d[1:] for d in test]
ytest = [d[0] for d in test]
results = []
for d in xtest:
    t = w[0]
    for i in range(1, len(w)):
        t += w[i]*d[i-1]
    if t > 0:
        results.append(1)    
    else:
        results.append(2)

t = numpy.array(ytest)
a = numpy.array(results)
cnf_matrix = confusion_matrix(t, a)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[1,2],title='Confusion matrix, without normalization')
plt.show()


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

xtrain = [d[1:] for d in data[:91]]
ytrain = [d[0] for d in data[:91]]
gnb.fit(xtrain, ytrain)
y_pred = gnb.predict(xtest)
b = numpy.array(y_pred)