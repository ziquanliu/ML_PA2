import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

data=scio.loadmat('cluster_data.mat')
A_X=data['dataA_X']
A_Y=data['dataA_Y']


#In this program, we set cluster number as 3,i.e. K=3

#K-means

def squ_Euc_dist(x,mean_value):
    return np.sum(np.square(x-mean_value))


K=4
X=np.array(A_X)
dim=X.shape[0]
num=X.shape[1]
#initialize miu_j
miu=np.zeros((dim,K))
for i in range(K):
    miu[:,i]=X[:,int(num*np.random.rand())]


#initialize z_ij
z=np.zeros((num,K))
for i in range(num):
    min_distance=squ_Euc_dist(X[:,i],miu[:,0])
    ind=0
    for j in range(1,K):
        distance=squ_Euc_dist(X[:,i],miu[:,j])
        if distance<min_distance:
            min_distance=distance
            ind=j
    z[i,ind]=1


#plot
l_type=['r.','b.','g.','y.']
def cluster_plt(line_type,z,X,num,K):
    f1 = plt.figure()
    for j in range(K):
        num_k = int(np.sum(z[:, j]))
        X_temp = np.zeros((dim, num_k))
        ind_temp = 0
        for i in range(num):
            if z[i, j] == 1:
                X_temp[:, ind_temp] = X[:, i]
                ind_temp += 1
        plt.plot(X_temp[0, :], X_temp[1, :], line_type[j])
    plt.show()

#-----------------
cluster_plt(l_type,z,X,num,K)

#iteration
center_shift=100
min_shift=10**-10
iter_n=0
miu_new=np.zeros((dim,K))

while center_shift>min_shift:
    print 'iteration number',iter_n
    iter_n+=1
    for j in range(K):
        denom=0
        for i in range(num):
            denom+=float(z[i,j])*X[:,i]
        miu_new[:,j]=denom/float(np.sum(z[:,j]))
    center_shift=np.sum(np.absolute(miu_new-miu))
    print 'center shift:',center_shift
    miu=miu_new.copy()
    z = np.zeros((num, K))
    for i in range(num):
        min_distance = squ_Euc_dist(X[:, i], miu[:, 0])
        ind = 0
        for j in range(1, K):
            distance = squ_Euc_dist(X[:, i], miu[:, j])
            if distance < min_distance:
                min_distance = distance
                ind = j
        z[i, ind] = 1
    cluster_plt(l_type,z,X,num,K)








#EM-GMM



#Mean-shift