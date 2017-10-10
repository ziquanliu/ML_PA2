import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import cluster_cls as clst

data=scio.loadmat('cluster_data.mat')
A_X=data['dataA_X']
A_Y=data['dataA_Y']

K=4
X=np.array(A_X)
l_type=['r.','b.','g.','y.']
true_label=np.array(A_Y)


#initialize parameters
dim=X.shape[0]
num=X.shape[1]
miu=np.zeros((dim,K))
Sigma=[]
for i in range(K):
    miu[:,i]=X[:,int(num*np.random.rand())]
    Sigma.append(np.eye(dim,dim)+np.random.rand(dim,dim))




