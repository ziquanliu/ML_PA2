import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import cluster_cls as clst

data=scio.loadmat('cluster_data.mat')
A_X=data['dataC_X']
A_Y=data['dataC_Y']

K=4
X=np.array(A_X)
l_type=['r.','b.','g.','y.']
true_label=np.array(A_Y)

clusters=clst.cluster(X,K,l_type,true_label)
h=np.zeros((1000,1))
for i in range(1000):
    h[i,0]=1+float(i)/1000.0
for i in range(1000)
    z_ms=clusters.mean_shift(h[i,0])
