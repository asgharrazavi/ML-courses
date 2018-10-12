import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import scipy.optimize as op
import copy

#-----------loading data----------------
data = io.loadmat('ex6/ex6data1.mat')
print data.keys()
y = data['y']					#shape : (5000,1)
x = data['X']					#shape : (5000, 400)
print "y.shape, x.shape:", y.shape, x.shape
#--------------------------------------

#----------plot data-------------------
plt.figure(figsize=(7,5))
pos = (y[:,0] == 1)
neg = (y[:,0] == 0)
plt.plot(x[:,0][pos],x[:,1][pos],'k+')
plt.plot(x[:,0][neg],x[:,1][neg],'yo')
plt.show()
#--------------------------------------


