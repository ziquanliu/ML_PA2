import numpy as np
import matplotlib.pyplot as plt


def squ_Euc_dist(x, mean_value):
    return np.sum(np.square(x - mean_value))


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



