import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import scipy.optimize as op
import copy


#-----------loading data----------------
data = io.loadmat('ex3data1.mat')
print data.keys()
y = data['y']
X = data['X']
print "data['y'].shape, data['X'].shape:", data['y'].shape, data['X'].shape
#--------------------------------------

#-----------plot some of the data------
def plot():
    m = len(y)
    rand_x = np.random.choice(m,100)
    xx = X[rand_x,:]
    for i in range(100):
    	plt.subplot(10,10,i+1)
    	img = np.reshape(xx[i],(20,20))
    	plt.imshow(np.rot90(img))
    	plt.xlim([0,20])
    	plt.ylim([0,20])
    	plt.xticks([])
    	plt.yticks([])
    plt.show()
#-------------------------------------

#-------------OneVsAll--------------------------------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def _costFunctionReg(theta, X, y,lambdaa):       # set regularization parameter lambda to 1
    m = float(len(y))
    h_theta = sigmoid(np.dot(theta,X.T))
    h, reg = h_theta, float(lambdaa)
    # we don't regularize theta[0], 
    J = (1.0/m) * np.sum( -1.0*y * np.log(h_theta) - (1-y) * np.log((1-h_theta)) ) + (float(lambdaa)/(2*m)) * np.sum(theta[1:]**2)
    ## very strangly this won't work
#    J = (1.0/m) * np.sum( -y * np.log(h_theta) - (1-y) * np.log((1-h_theta)) ) + (float(lambdaa)/(2*m)) * np.sum(theta[1:]**2)
#    J = -(1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (reg/(2*m))*np.sum(np.square(theta[1:]))
    if np.isnan(J):
        return(np.inf)
    return J

def _gradientReg(theta, X, y,lambdaa):
    grad = np.zeros(len(theta))
    n_parms = len(theta)
    m = float(len(y))
    h_theta = sigmoid(np.dot(theta,X.T))
    for i in range(n_parms):
        if i == 0: grad[i] = (1.0/m) * np.sum((h_theta - y)*X[:,i])
        else: grad[i] = ( (1.0/m) * np.sum((h_theta - y)*X[:,i]) ) + lambdaa * theta[i] / float(m)
    return grad


def graid_dec(theta,X,y,alpha,lambdaa,n_iters):
    print "init cost:", __costFunctionReg(theta, X, y,lambdaa)[0]
    for i in range(n_iters):
  	cost = __costFunctionReg(theta, X, y,lambdaa)
	theta = theta - alpha * cost[1]
    print "final cost:", cost[0]
    return theta

def oneVsAll(X, y, num_labels, lambdaa, alpha, n_iters):		#X: (#of examples,#of parameters)
    m = X.shape[0]
    n = X.shape[1]					
    all_theta = np.zeros((num_labels, n ))
    for i in range(num_labels):
	yy = io.loadmat('ex3/ex3data1.mat')['y'][:,0] 
	ind_0 = (y != i+1)
	ind_1 = (y == i+1)
   	yy[ind_1] = 1
   	yy[ind_0] = 0
	print i+1, yy, len(np.where(yy == 1)[0]), yy.shape
 	init_theta = all_theta[i]
	out = op.fmin_bfgs(_costFunctionReg,init_theta,_gradientReg,args=(X,yy,lambdaa),full_output=True,maxiter=500)
#	out = op.minimize(_costFunctionReg,init_theta,args=(X,yy,lambdaa),method=None,jac=_gradientReg,options={'disp':True})
#	out = graid_dec(init_theta,X,yy,alpha,lambdaa,n_iters)
	all_theta[i] = out[0]
    return all_theta

X = np.c_[np.ones(X.shape[0]),X]
lambdaa = 0.1
alpha = 0.00001
n_iters = 20
num_labels = len(np.unique(y))
init_theta = np.zeros(X.shape[1])
y = y[:,0]
print "X.shape, y.shape:", X.shape, y.shape
all_theta = oneVsAll(X, y, num_labels, lambdaa, alpha, n_iters)
np.savetxt('all_theta.txt',all_theta)
#----------------------------------------------------------------------------------------------------------------------------

#------------Predict-------------------
all_theta = np.loadtxt('all_theta.txt')
def predictOneVsAll(all_theta, X):
    predicts = np.zeros((X.shape[0]),dtype=int)
    for i in range(all_theta.shape[0]):
        h_theta = sigmoid(np.dot(all_theta[i],X.T))
	print h_theta.shape, len(np.where(h_theta >= 0.5)[0])
        predicts[h_theta >= 0.5] = i+1
    return predicts

p = predictOneVsAll(all_theta, X)
print y, "\n",p
print np.where(p == y.ravel())
print np.mean(p == y.ravel())*100
print "prediction accuracy: %f percent" %(100 *(len(y)-np.sum(abs(p-y)))/float(len(y)))
