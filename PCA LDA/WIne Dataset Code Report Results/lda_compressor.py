import csv
import numpy
import urllib
import random
import itertools
import numpy as np
from math import exp
from math import log
import scipy.optimize
from random import shuffle
import scipy.linalg as linalg
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

class LDA:
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
        xtrain1 = [f for f in features1[:n]]
        ytrain1 = [y for y in label1[:n]]
        xtrain2 = [f for f in features2[:n]]
        ytrain2 = [y for y in label2[:n]]
        xtrain = xtrain1 + xtrain2
        ytrain = ytrain1 + ytrain2

        mtrain1 = np.matrix(xtrain1)
        mean1 = mtrain1.mean(0)
        mtrain2 = np.matrix(xtrain2)
        mean2 = mtrain2.mean(0)
        mtrain = np.matrix(xtrain)
        mean = mtrain.mean(0)

        d = len(xtrain1[0])
        sca_mat1 = np.zeros((d,d))
        for i in range(len(xtrain1)):
            sample = np.array(xtrain1[i])
            sca_mat1 += (sample - mean1).T.dot((sample - mean1))

        d = len(xtrain2[0])
        sca_mat2 = np.zeros((d,d))
        for i in range(len(xtrain2)):
            sample = np.array(xtrain2[i])
            sca_mat2 += (sample - mean2).T.dot((sample - mean2))

        sw = (sca_mat1 + sca_mat2)
        sb = n*(mean1-mean).T.dot((mean1-mean)) + n*(mean2-mean).T.dot((mean2-mean))

        eig_val_sc, eig_vec_sc = np.linalg.eigh(np.linalg.pinv(sw).dot(sb))
        eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        return eig_pairs, mean

    def compressor(self, eig_pairs, features, label, deigen = 5):
        matrix_w = []
        for i in range(deigen):
            e = eig_pairs[i][1].tolist()
            l = [v[0] for v in e]
            matrix_w.append(l)
        matrix_w = np.matrix(matrix_w)
        matrix_w = matrix_w.T
        #print('Matrix W:\n', matrix_w)
        
        x = np.matrix(features)
        tx = x.dot(matrix_w)

        features = tx.tolist()
        data = [[label[i]] + features[i] for i in range(len(features))]
        return data, matrix_w           

    def classifier(self, data):
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

        features = features1 + features2
        xtest = [[1.0] + f for f in features]
        label = label1 + label2
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
        precision = sum(c)*1.0/len(c)
        print "Precicion :"+ str(precision)

        cnf_matrix = confusion_matrix(b, results)
        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=[1,2],title='Confusion matrix, without normalization')
        plt.show()
        return precision, weight