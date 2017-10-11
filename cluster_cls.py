import numpy as np
import matplotlib.pyplot as plt
import copy
import time


def squ_Euc_dist(x, mean_value):
    return np.sum(np.square(x - mean_value))


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

def cluster_plt(line_type,z,X,K):
    dim = X.shape[0]
    num = X.shape[1]
    f1 = plt.figure()
    z_t=z.copy()
    for i in range(num):
        max_index=np.argmax(z_t[i,:])
        z_t[i,:]=np.zeros((1,K))
        z_t[i,max_index]=1

    for j in range(K):
        num_k = int(np.sum(z_t[:, j]))
        X_temp = np.zeros((dim, num_k))
        ind_temp = 0
        for i in range(num):
            if z_t[i, j] == 1:
                X_temp[:, ind_temp] = X[:, i]
                ind_temp += 1
        plt.plot(X_temp[0, :], X_temp[1, :], line_type[j])
    plt.show()



class cluster(object):
    def __init__(self,X,K,l_type,true_label):
        self.X=X
        self.K=K
        self.line_type=l_type
        self.true_X=true_label



    def K_means(self):
        # initialize miu_j
        dim = self.X.shape[0]
        num = self.X.shape[1]
        miu = np.zeros((dim, self.K))
        for i in range(self.K):
            miu[:, i] = self.X[:, int(num * np.random.rand())]

        # initialize z_ij
        z = np.zeros((num, self.K))
        for i in range(num):
            min_distance = squ_Euc_dist(self.X[:, i], miu[:, 0])
            ind = 0
            for j in range(1, self.K):
                distance = squ_Euc_dist(self.X[:, i], miu[:, j])
                if distance < min_distance:
                    min_distance = distance
                    ind = j
            z[i, ind] = 1
        cluster_plt(self.line_type, z, self.X,self.K)

        # iteration
        center_shift = 100
        min_shift = 10 ** -10
        iter_n = 0
        miu_new = np.zeros((dim, self.K))

        while center_shift > min_shift:
            print 'iteration number', iter_n
            iter_n += 1
            for j in range(self.K):
                denom = 0
                for i in range(num):
                    denom += float(z[i, j]) * self.X[:, i]
                miu_new[:, j] = denom / float(np.sum(z[:, j]))
            center_shift = np.sum(np.absolute(miu_new - miu))
            print 'center shift:', center_shift
            miu = miu_new.copy()
            z = np.zeros((num, self.K))
            for i in range(num):
                min_distance = squ_Euc_dist(self.X[:, i], miu[:, 0])
                ind = 0
                for j in range(1, self.K):
                    distance = squ_Euc_dist(self.X[:, i], miu[:, j])
                    if distance < min_distance:
                        min_distance = distance
                        ind = j
                z[i, ind] = 1
            cluster_plt(self.line_type, z, self.X, self.K)
        return z


    def EM_GMM(self):
        dim = self.X.shape[0]
        num = self.X.shape[1]
        miu = np.zeros((dim, self.K))
        Sigma = []
        for i in range(self.K):
            miu[:, i] = self.X[:, int(num * np.random.rand())]  # initialize miu
            Sigma.append(gene_cov(dim))  # initialize Sigma

        pi = np.ones((1, self.K)) / float(self.K)  # initialize pi

        z = np.zeros((num, self.K))  # initialize z
        resid = 10
        MIN_RES = 10 ** -10
        pi_new = pi.copy()
        miu_new = miu.copy()
        Sigma_new = copy.copy(Sigma)
        iter_num = 0

        while resid > MIN_RES:

            print 'iteration number', iter_num
            iter_num += 1
            for i in range(num):
                compnt = np.zeros((1, self.K))
                for j in range(self.K):
                    compnt[0, j] = pi[0, j] * cal_gaussian(self.X[:, i], miu[:, j], Sigma[j])
                for j in range(self.K):
                    z[i, j] = compnt[0, j] / np.sum(compnt)
            # print z
            cluster_plt(self.line_type, z, self.X, self.K)
            # print np.sum(z,0)
            N = np.sum(z, 0).reshape((1, self.K))
            for j in range(self.K):
                pi_new[0, j] = N[0, j] / num
                miu_new[:, j] = cal_new_miu(j, N[0, j], z, self.X)
                Sigma_new[j] = cal_new_cov(j, N[0, j], z, self.X, miu_new[:, j])
                # break
            # break
            resid = np.sum(np.abs(pi - pi_new)) + np.sum(np.abs(miu - miu_new))
            print 'resid', resid
            pi = pi_new.copy()
            miu = miu_new.copy()
            Sigma = copy.copy(Sigma_new)
        return z



