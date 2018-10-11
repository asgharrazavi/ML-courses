import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import scipy.optimize as op
import copy
from matplotlib.colors import LogNorm

#-----------loading data----------------
data = io.loadmat('ex4/ex4data1.mat')
print data.keys()
y = data['y']                                   #shape : (5000,1)
x = data['X']                                   #shape : (5000, 400)
print "y.shape, x.shape:", y.shape, x.shape
#--------------------------------------

plt.figure(figsize=(20,15))
for i in range(x.shape[1]):
    if abs(np.min(x[:,i])-np.max(x[:,i])) > 1: print i, np.min(x[:,i]), np.max(x[:,i]), np.std(x[:,i])
    h = np.histogram(x[:,i],bins=100)
    plt.subplot(20,20,i+1);plt.plot(h[1][0:-1],h[0])
    plt.yticks([])
    plt.xticks([])
    plt.title(i)

plt.show()
