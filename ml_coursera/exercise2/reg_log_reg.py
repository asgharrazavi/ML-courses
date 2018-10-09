import numpy as np
import matplotlib.pyplot as plt

#---------------loading data-----------------------
def plot(x,y):
    admitted = (y == 1)
    not_admitted = (y == 0)
    plt.plot(x[:,0][admitted],x[:,1][admitted],'k+')
    plt.plot(x[:,0][not_admitted],x[:,1][not_admitted],'yo')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(('y = 1','y = 0'))
data = np.loadtxt('ex2/ex2data2.txt',delimiter=',')
x = data[:,0:2]
y = data[:,2]
#plot(x,y)
#plt.show()
#-------------------------------------------------

#------------Map features-------------
def mapFeature(x):
    data = []
    for i in range(1,7):
	for j in range(i+1):
	    data.append((x[:,0]**(i-j)*x[:,1]**j))
#	    print i-j,j
    return np.array(data).T
xx = mapFeature(x)
X = np.c_[np.ones(len(y)),xx]
print X.shape
#------------------------------------

#----------sigmoid(x)-----------------
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
#print sigmoid(0) 
#print sigmoid(1) 
#print sigmoid(3489) 
#print sigmoid(-39) 
#print sigmoid(np.random.random(10))
#--------------------------------------

#-------------Cost function-------------------------
def costFunctionReg(theta, X, y,lambdaa):	# set regularization parameter lambda to 1
    grad = np.zeros(len(theta))
    n_parms = len(theta)
    m = float(len(y))
    h_theta = sigmoid(np.dot(theta,X.T))
    J = (1.0/m) * np.sum( -y * np.log(h_theta) - (1-y) * np.log((1-h_theta)) ) + (float(lambdaa)/(2*m)) * np.sum(theta**2)
    for i in range(n_parms):   
	if i == 0: grad[i] = (1.0/m) * np.sum((h_theta - y)*X[:,i])
	else: grad[i] = ( (1.0/m) * np.sum((h_theta - y)*X[:,i]) ) + lambdaa * theta[i] / float(m)
    return J, grad

n = X.shape[1]
theta = np.zeros(n)
alpha, lambdaa = 1, 1
J, grad = costFunctionReg(theta, X, y,lambdaa)
print "gradients:", grad
print "cost function:", J
#---------------------------------------------------

#----------Gradient Descent------------------------
def gradient_descent(theta,X,y,n_iter,alpha,lambdaa):
    costs = np.zeros(n_iter)
    for i in range(n_iter):
        h_theta = sigmoid(np.dot(theta,X.T))
        cost = costFunctionReg(theta, X, y,lambdaa)
	costs[i] = cost[0]
   	theta = theta - alpha * cost[1]
        if i%2000==0 : print costs[i],theta[0:4]
    return theta, costs

n_iter = 5000
alpha = 1
lambdaa = 1.0
n = X.shape[1]
init_theta = np.ones(n)
theta, costs = gradient_descent(theta,X,y,n_iter,alpha,lambdaa)
print "cost function:", costs[-1]
print "thetas:", theta
#-------------------------------------------------

#----------advanced optimization algorithms---------------------------------------------------------------------------
from scipy.optimize import fmin_bfgs
def func(theta,X,y,lambdaa):
    return costFunctionReg(theta, X, y,lambdaa)[0]    
def gradient(theta,X,y,lambdaa):
    return costFunctionReg(theta, X, y,lambdaa)[1]
init_theta = np.zeros(n)
lambdaa = 1.0
out = fmin_bfgs(func,init_theta,gradient,args=(X,y,lambdaa),full_output=True)
theta = out[0]
min_cost = out[1]
min_gradients = out[3]   
print "\n\nresults from advanced optimization algorithm:\n"
print "theta:",theta
print "min cost:", min_cost
#print "mingradient:", min_gradients
#----------------------------------------------------------------------------------------------------------------------

#---------plotting decision boundray-------------
def plotDecisionBoundary(theta, X, y):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u),len(v)))
    for i in range(len(u)):
    	    uv = np.ones((len(u),2))
    	    uv[:,0], uv[:,1] = np.ones(len(u))*u[i], v
    	    zz = mapFeature(uv)   
    	    zz = np.c_[np.ones(len(u)),zz] 
    	    z[i] = np.dot(zz,theta)
    print z.shape
    plt.contour(u,v,z,1)		# plot only 1 contour
    plot(x,y)

plotDecisionBoundary(theta, X, y)
plt.show()
#-------------------------------------------------    

#----------predict------------------------------------------------------------------------
x0, x1, x2 = 1, 45, 85
theta = [-25.161, 0.206, 0.201]
h_theta = sigmoid(np.dot(theta,[x0,x1,x2]))
print h_theta
def predict(theta, X):
     h_theta = sigmoid(np.dot(theta,X.T))
     h_theta[h_theta >= 0.5] = 1
     h_theta[h_theta <  0.5] = 0
     return h_theta
p = predict(theta, X)
print y, "\n",p
print "prediction accuracy: %f percent" %(100 *(len(y)-np.sum(abs(p-y)))/float(len(y)))
#-----------------------------------------------------------------------------------------



