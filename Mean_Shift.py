import numpy as np
import scipy.io as scio
import pickle
import matplotlib.pyplot as plt
import cluster_cls as clst
import copy

data=scio.loadmat('cluster_data.mat')
A_X=data['dataA_X']
A_Y=data['dataA_Y']

X=np.array(A_X)
l_type=['r.','b.','g.','y.']
true_label=np.array(A_Y)
h=1.0 #bandwidth

def cal_gaussian(x,miu,Cov):
    exp_f=-(x-miu).transpose().dot(np.linalg.inv(Cov)).dot(x-miu)/2.0
    return np.exp(exp_f)/np.sqrt(np.linalg.det(Cov))


dim=X.shape[0]
num=X.shape[1]
Sigma=h**2*np.eye(dim,dim)
shift_ind=np.ones((1,num))
MIN_RES=10**-10
x_mean=X.copy()
iter_num=0
while np.sum(shift_ind)>0:
    print 'iteration number',iter_num
    iter_num+=1
    for i in range(num):
        if shift_ind[:, i] == 0: continue
        x_old = x_mean[:, i].reshape(dim, 1)
        x_new = update_mean(x_old, X, Sigma)
        if np.sum(np.abs(x_old - x_new)) < MIN_RES:
            shift_ind[:, i] = 0
        else:
            x_mean[:, i] = x_new.reshape((dim))
    if iter_num>10**4:
        break


z,K=clst.ms_find_cluster(h,x_mean,X)
