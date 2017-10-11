import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import cluster_cls as clst

data=scio.loadmat('cluster_data.mat')
A_X=data['dataB_X']
A_Y=data['dataB_Y']

K=4
X=np.array(A_X)
l_type=['r.','b.','g.','y.']
true_label=np.array(A_Y)

clusters=clst.cluster(X,K,l_type,true_label)
z_K_means=clusters.K_means()
z_EM=clusters.EM_GMM()
