import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import scipy.optimize as op
import copy


#-----------loading data----------------
data = io.loadmat('ex3data1.mat')
print data.keys()
y = data['y']
x = data['X']
X = np.c_[np.ones(x.shape[0]),x]
print "data['y'].shape, data['X'].shape:", data['y'].shape, data['X'].shape
#--------------------------------------

#--------------load neural network parameters----------
data = io.loadmat('ex3weights.mat')
Theta1 = data['Theta1']
Theta2 = data['Theta2']
print "Theta1.shape, Theta2.shape:", Theta1.shape, Theta2.shape
#------------------------------------------------------

#---------------predicting labels from neural network parameters------------
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def predict(Theta1, Theta2, X):			#Theta1: (25,401), Theta2: (10,26), X: (5000,401)
    nn_parms = [Theta1, Theta2]
    for i in range(len(nn_parms)):
        if i == 0:  zz = np.dot(nn_parms[i],X.T).T ; aa = sigmoid(zz)
        else: aa = np.c_[np.ones(aa.shape[0]),aa]; zz = np.dot(nn_parms[i],aa.T).T ; aa = sigmoid(zz)
#    a1 = X
#    z2 = np.dot(Theta1,a1.T).T
#    a2 = sigmoid(z2)
#    a2 = np.c_[np.ones(a2.shape[0]),a2]
#    z3 = np.dot(Theta2,a2.T).T
#    a3 = sigmoid(z3)
    a3 = np.rint(aa)				#a3.shape: (5000,10)
    p = np.zeros(X.shape[0],dtype=int)
    for i in range(len(p)):
	w = np.where(a3[i]==1)[0]
        if len(w) > 0: p[i] = w[0]+1
	else: pass
    return p
p = predict(Theta1, Theta2, X)
print "prediction accuracy: %f percent:" %(np.mean(p == y.ravel())*100)
#-----------------------------------------------------------------------------

#--------------plotting predictions-----------------
def plot(X,y):
    m = len(y)
    rand_x = np.random.choice(m,100)
    xx = X[rand_x,:]
    p = predict(Theta1, Theta2, xx)
    for i in range(100):
        plt.subplot(10,10,i+1)
        try: img = np.reshape(xx[i],(20,20))
        except: img = np.reshape(xx[i][1:],(20,20))
        plt.imshow(np.rot90(img))
        plt.xlim([0,20])
        plt.ylim([0,20])
        plt.xticks([])
        plt.yticks([])
	plt.text(20,20,p[i])
    plt.show()
plot(X,y)
#-------------------------------------
