import numpy as np
import matplotlib.pyplot as plt

#-------------------------------#
#       Linear Regression   	#
#-------------------------------#

def featureNormalize(X):
    av = np.mean(X,axis=0)
    sd = np.std(X,axis=0)
    xx = np.zeros(X.shape)
    xx[:,0] = (X[:,0] - av[0]) / float(sd[0])
    xx[:,1] = (X[:,1] - av[1]) / float(sd[1])
#    print "feature one range:", np.min(xx[:,0]) - np.max(xx[:,0])
#    print "feature two range:", np.min(xx[:,1]) - np.max(xx[:,1])
    return xx
    
#--------------data--------------------------------
data = np.loadtxt('ex1/ex1data2.txt',delimiter=',')
X = data[:,0:2]
y = data[:,2]
m = len(y)
#--------------------------------------------------

#----feature normalization for faster gradient descent-------
f_Norm = featureNormalize(X)
#------------------------------------------------------------

#--------------------------
# Add intercept term to X        tet_0 * X0 + tet_1 * X1 + tet_2 * X2  (we add X0 (all 1))
X2 = np.ones((X.shape[0],3))
X2[:,1] = f_Norm[:,0]
X2[:,2] = f_Norm[:,1]
#----------------------------

#-------Cost function-----------------
def computeCostMulti(X, y, theta):
    h_theta = np.dot(theta,X.T)
    j_theta = (0.5/m) * np.sum((h_theta - y)**2)
    return j_theta
theta = np.zeros(3)
X = X2
print "cost function:", computeCostMulti(X,y,theta)

#-------Gradient Descent--------------
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    
    def _cost(X, y, theta):
        m = len(y)
	h_theta = np.dot(theta,X.T)
 	j_theta = np.zeros(len(theta))
	for i in range(len(theta)):
    	    j_theta[i] = (alpha/m) * np.sum((h_theta - y)*X[:,i])
    	return j_theta
    
    print "theta_0 \t theta_1 \t theta_2 \t cost "
    costs = []
    for i in range(num_iters):
        new_cost = _cost(X, y, theta)
        theta = theta - new_cost
        cc = computeCostMulti(X, y, theta)
  	costs.append(cc)
        if i%100==0: print "%1.3f \t  %1.3f \t  %1.3f \t  %s" %(theta[0], theta[1], theta[2], cc)
    
    return theta, costs

alpha = 0.01
num_iters = 400
#theta, costs = gradientDescentMulti(X, y, theta, alpha, num_iters)
#plt.plot(costs)
#plt.show()

#-----------finding better learning rates (alpha values) ----------
alphas = [0.3,0.1,0.03,0.01]
for alpha in alphas:
    theta = np.zeros(X.shape[1])
    theta, costs = gradientDescentMulti(X, y, theta, alpha, num_iters)
    plt.plot(costs)
plt.legend(alphas)
#plt.show()
#------------------------------------------------------------------

#------------Normal Equations--------------------------------------
invX = np.linalg.inv(np.dot(X.T,X))
print "invX.shape", invX.shape
invXX = np.dot(invX,X.T)
print "invXX.shape", invXX.shape
theta = np.dot(invXX,y)
print "theta:", theta

