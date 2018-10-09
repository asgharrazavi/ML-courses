import numpy as np
import matplotlib.pyplot as plt

#---------------loading data-----------------------
def plot(x,y):
    admitted = (y == 1)
    not_admitted = (y == 0)
    plt.plot(x[:,0][admitted],x[:,1][admitted],'k+')
    plt.plot(x[:,0][not_admitted],x[:,1][not_admitted],'yo')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(('admitted','not_admitted'))
data = np.loadtxt('ex2/ex2data1.txt',delimiter=',')
x = data[:,0:2]
y = data[:,2]
#plot(x,y)
#plt.show()
#-------------------------------------------------

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
def costFunction(theta, X, y):				#h_theta is the sigmoid function
    grad = np.zeros(len(theta))
    J = 0
    n_parms = X.shape[1]
    m = len(y)
    h_theta = sigmoid(np.dot(theta,X.T))
    J = (1.0/m) * np.sum( -y * np.log(h_theta) - (1-y) * np.log((1-h_theta)) )
    for i in range(n_parms):   
	grad[i] = (1.0/m) * np.sum((h_theta - y)*X[:,i])
    return J, grad

n = x.shape[1]
theta = np.zeros(n+1)
X = np.c_[np.ones(len(y)),x]
J, grad = costFunction(theta, X, y)
print "gradients:", grad
print "cost function:", J
test_theta = [-24, 0.2, 0.2]
#test_theta = [-10, 0.2, 0.2]
J, grad = costFunction(test_theta, X, y)
print "gradients:", grad
print "cost function:", J
#---------------------------------------------------

#----------Gradient Descent------------------------
def gradient_descent(theta,X,y,n_iter,alpha):
    costs = np.zeros(n_iter)
    for i in range(n_iter):
        h_theta = sigmoid(np.dot(theta,X.T))
	costs[i] = costFunction(theta, X, y)[0]
        for j in range(len(theta)):
	    theta[j] = theta[j] - alpha * (np.sum((h_theta - y)*X[:,j]))
	if i%2000==0 : print costs[i],theta
    return theta, costs

#n_iter = 400000
#alpha = 0.000011
#theta = test_theta
#theta = np.zeros(3)
#theta, costs = gradient_descent(theta,X,y,n_iter,alpha)
#print "cost function:", costs[-1]
#print "thetas:", theta
#-------------------------------------------------

#---------plotting decision boundray-------------
def plotDecisionBoundary(theta, X, y):
#    plot_x = [np.min(X[:,2]-2), np.max(X[:,2]+2)]
#    plot_y = [(-1.0/theta[2])*(theta[1]*plot_x[0]+theta[0]), (-1.0/theta[2])*(theta[1]*plot_x[1]+theta[0])]
    plot_x = np.array([np.min(X[:,2]-2), np.max(X[:,2]+2)])
    x1 = plot_x
    theta_0 = np.array([theta[0] for i in range(len(x1))])
    x2 = -(theta[1]*x1 + theta_0) / float(theta[2])    #      theta_1 * x1 + theta_2 * x2 + theta_0 = 0
    plot_y = x2
    plot(X[:,1:3],y)
    plt.plot(plot_x,plot_y,lw=2)			
   
theta = [-25.161, 0.206, 0.201]
plotDecisionBoundary(theta, X, y)
plt.show()
    
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



