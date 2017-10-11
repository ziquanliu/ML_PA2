import numpy as np
import scipy.io as scio
import pickle
import time
import matplotlib.pyplot as plt
import cluster_cls as clst
import copy

data=scio.loadmat('cluster_data.mat')
A_X=data['dataA_X']
A_Y=data['dataA_Y']
h=2.0
X_Mean=pickle.load(open('h_'+str(h)+'_mean_shift_result.txt','rb'))
X=np.array(A_X)
dim=X.shape[0]
num=X.shape[1]
cluster_cen=[X_Mean[:,0].reshape((dim,1))]
MIN_MEAN_DEV=h
num_cen=1
for i in range(num):
    temp=np.zeros((num_cen,1))
    for j in range(num_cen):
        temp[j,0]=np.sum(np.absolute(X_Mean[:,i].reshape((dim,1))-cluster_cen[j]))
    if np.min(temp)>MIN_MEAN_DEV:
        cluster_cen.append(X_Mean[:,i].reshape((dim,1)))
        num_cen+=1

z=np.zeros((num,num_cen))
for i in range(num):
    temp = np.zeros((num_cen, 1))
    for j in range(num_cen):
        temp[j, 0] = np.sum(np.absolute(X_Mean[:, i].reshape((dim, 1)) - cluster_cen[j]))
    z[i,np.argmin(temp)]=1


print z
print num_cen
print cluster_cen
