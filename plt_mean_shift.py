import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

import pickle

data=scio.loadmat('cluster_data.mat')
A_X=data['dataC_X']
A_Y=data['dataC_Y']

K=4
X=np.array(A_X)
line_type=['r.','b.','g.','y.']

x_mean=pickle.load(open('data/h_1.7_mean_shift_mean.txt','rb'))
z=pickle.load(open('data/h_1.7_mean_shift_z.txt','rb'))

dim = X.shape[0]
num = X.shape[1]
f1 = plt.figure()
z_t = z.copy()
for i in range(num):
    max_index = np.argmax(z_t[i, :])
    z_t[i, :] = np.zeros((1, K))
    z_t[i, max_index] = 1

for j in range(K):
    num_k = int(np.sum(z_t[:, j]))
    X_temp = np.zeros((dim, num_k))
    ind_temp = 0
    for i in range(num):
        if z_t[i, j] == 1:
            X_temp[:, ind_temp] = X[:, i]
            ind_temp += 1
    plt.plot(X_temp[0, :], X_temp[1, :], line_type[j])
plt.plot(x_mean[0,:],x_mean[1,:],'kx')
plt.show()