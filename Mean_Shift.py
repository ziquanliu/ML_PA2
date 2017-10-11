import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import cluster_cls as clst
import copy

data=scio.loadmat('cluster_data.mat')
A_X=data['dataA_X']
A_Y=data['dataA_Y']

K=4
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


def update_mean(x_old,DS,Sigma):
    nominator=np.zeros(x_old.shape)
    denom=0.0
    num_DS=X.shape[1]
    for i in range(num_DS):
        g_kern=cal_gaussian(DS[:,i].reshape(dim,1),x_old,Sigma)
        nominator+=(DS[:,i].reshape(dim,1))*g_kern
        denom+=g_kern
    return nominator/denom

for i in range(num):
    x_old=X[:,i].reshape(dim,1)
    x_new=update_mean(x_old,X,Sigma)
