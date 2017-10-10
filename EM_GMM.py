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

def gene_cov(dim):
    cov=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(i,dim):
            if i==j:
                cov[i,j]=1.0+np.random.rand()
            else:
                cov[i,j]=-1.0+np.random.rand()*2
                cov[j,i]=cov[i,j]
    return cov


#initialize parameters
dim=X.shape[0]
num=X.shape[1]
miu=np.zeros((dim,K))
Sigma=[]
for i in range(K):
    miu[:,i]=X[:,int(num*np.random.rand())]
    Sigma.append(gene_cov(dim))
print Sigma[0]


#Gaussian distribution
def cal_gaussian(x,miu,Cov):
    exp_f=-(x-miu).transpose().dot(np.linalg.inv(Cov)).dot(x-miu)/2.0
    return np.exp(exp_f)/np.sqrt(np.linalg.det(Cov))

print cal_gaussian(miu[:,0]+2,miu[:,0],Sigma[0])

