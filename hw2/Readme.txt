#Run following inside the unzipped directory

-------------------------- AdaBoost ----------------------------------
#To execute random forest for MNIST
import random_forest as rf
import numpy as np
data = rf.parseData('train.csv')
x = int(.75*len(data))
train = np.array(data[:x])
val = np.array(data[x:])
M = 50  #Number of Features
ntrees = 80 #Number of Trees
bootstrap_size = 5000 #BootStrap Size 
max_depth = 5 #Depth of trees
rfc = rf.Random_Forest(train,ntrees,bootstrap_size,M,max_depth)
rfc.create_trees()
acc, true, pred = rfc.predictDataset(val)

#To execute AdaBoost for MNIST Data
import random_forest_boost as rfb
import numpy as np
data = rfb.parseData('train.csv')
x = int(.75*len(data))
train = np.array(data[:x])
val = np.array(data[x:])
ntrees = 100
max_depth = 10
rfc = rfb.Random_Forest(train,ntrees, max_depth)
rfc.create_trees()
a,t,p = rfc.predictDataset(val)

-------------------------- CoverType ----------------------------------

#To execute random forest for Covertype Data
import random_forest as rf
import numpy as np
t = rf.parseData('covtype.data')
data = [[d[-1]] + d[:-1] for d in t]
x = int(.75*len(data))
train = np.array(data[:x])
val = np.array(data[x:])
M = 25  #Number of Features
ntrees = 80 #Number of Trees
bootstrap_size = 5000 #BootStrap Size 
max_depth = 5 #Depth of trees
rfc = rf.Random_Forest(train,ntrees,bootstrap_size,M,max_depth)
rfc.create_trees()
acc, true, pred = rfc.predictDataset(val)

#To execute AdaBoost for Covertype Data
import random_forest_boost as rfb
import numpy as np
t = rf.parseData('covtype.data')
data = [[d[-1]] + d[:-1] for d in t]
x = int(.75*len(data))
train = np.array(data[:x])
val = np.array(data[x:])
ntrees = 100
max_depth = 10
rfc = rfb.Random_Forest(train,ntrees, max_depth)
rfc.create_trees()
a,t,p = rfc.predictDataset(val)