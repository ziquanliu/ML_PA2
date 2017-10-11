import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import cluster_cls as clst
import copy

data=scio.loadmat('cluster_data.mat')
A_X=data['dataB_X']
A_Y=data['dataB_Y']




#Gaussian distribution
def cal_gaussian(x,miu,Cov):
    exp_f=-(x-miu).transpose().dot(np.linalg.inv(Cov)).dot(x-miu)/2.0
    return np.exp(exp_f)/np.sqrt(np.linalg.det(Cov))

def gene_cov(dim):
    cov=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(i,dim):
            if i==j:
                cov[i,j]=5.0+np.random.rand()
            else:
                cov[i,j]=-5.0+np.random.rand()*10
                cov[j,i]=cov[i,j]
    return cov


def cal_new_miu(j,N,z,X):
    sam_num=X.shape[1]
    s=np.zeros(X[:,0].shape)
    for i in range(sam_num):
        s+=z[i,j]*X[:,i]
    return s/N


def cal_new_cov(j,N,z,X,miu_j):
    sam_num=X.shape[1]
    sam_dim=X.shape[0]
    Sig_s=np.zeros((sam_dim,sam_dim))
    for i in range(sam_num):
        temp=(X[:,i]-miu_j).reshape(2,1)
        Sig_s+=z[i,j]*temp.dot(temp.transpose())
    return Sig_s/N


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
    miu[:,i]=X[:,int(num*np.random.rand())]#initialize miu
    Sigma.append(gene_cov(dim))#initialize Sigma

pi=np.ones((1,K))/float(K) #initialize pi

z=np.zeros((num,K))#initialize z
resid=10
MIN_RES=10**-10
pi_new=pi.copy()
miu_new=miu.copy()
Sigma_new=copy.copy(Sigma)
iter_num=0

while resid>MIN_RES:

    print 'iteration number',iter_num
    iter_num+=1
    for i in range(num):
        compnt = np.zeros((1, K))
        for j in range(K):
            compnt[0, j] = pi[0, j] * cal_gaussian(X[:, i], miu[:, j], Sigma[j])
        for j in range(K):
            z[i, j] = compnt[0, j] / np.sum(compnt)
    #print z
    clst.cluster_plt(l_type,z,X,K)
    #print np.sum(z,0)
    N=np.sum(z,0).reshape((1,K))
    for j in range(K):
        pi_new[0,j]=N[0,j]/num
        miu_new[:,j]=cal_new_miu(j,N[0,j],z,X)
        Sigma_new[j]=cal_new_cov(j,N[0,j],z,X,miu_new[:,j])
        #break
    #break
    resid=np.sum(np.abs(pi-pi_new))+np.sum(np.abs(miu-miu_new))
    print 'resid',resid
    pi=pi_new.copy()
    miu=miu_new.copy()
    Sigma=copy.copy(Sigma_new)



#E-step



#M-step


