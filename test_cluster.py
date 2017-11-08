import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import cluster_cls as clst

data=scio.loadmat('cluster_data.mat')
A_X=data['dataC_X']
A_Y=data['dataC_Y']

K=8
X=np.array(A_X)
l_type=['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'w.']
true_label=np.array(A_Y)
h=1.703
h_l=[2.1,2.2,2.4,2.6,2.8]
dim=X.shape[0]
num=X.shape[1]
clusters=clst.cluster(X,K,l_type,true_label)
#z_K_means,c_K_means=clusters.K_means()
#Y_EM,z_EM,c_EM=clusters.EM_GMM()

for h in h_l:
    z_ms, c_ms = clusters.mean_shift(h)
    clst.plt_save(l_type, z_ms, X, 'Mean_Shift_C', c_ms, h)

#z_ms,c_ms=clusters.mean_shift(h)

#clst.cluster_plt(l_type,z_ms,X,K,'mean_shift',0)
#clst.plt_save(l_type,z_K_means,X,K,'K_Means_C',c_K_means)
#clst.plt_save(l_type,z_EM,X,K,'EM_GMM_C',c_EM)
#clst.plt_save(l_type,z_ms,X,K,'Mean_Shift_C',c_ms,h)
#z_true=np.zeros((num,K))
#for i in range(num):
#    z_true[i,A_Y[0,i]-1]=1


#clst.cluster_plt(l_type,A_Y,A_X,K)