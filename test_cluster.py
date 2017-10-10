import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import cluster_cls as clst

data=scio.loadmat('PA2-cluster-data/cluster_data.mat')
A_X=data['dataA_X']
A_Y=data['dataA_Y']

K=4
X=np.array(A_X)
l_type=['r.','b.','g.','y.']
true_label=np.array(A_Y)

clusters=clst.cluster(X,K,l_type,true_label)
z=clusters.K_means()