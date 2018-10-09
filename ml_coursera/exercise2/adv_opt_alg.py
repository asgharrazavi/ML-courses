import numpy as np
from scipy.optimize import fmin_bfgs

def costfunction(theta):
    jval = (theta[0] - 5)**2 + (theta[1] - 5)**2
    gradient = np.zeros(2)
    gradient[0] = 2 * (theta[0]-5)
    gradient[1] = 2 * (theta[1]-5)
    return jval, gradient

def func(theta):
    jval = (theta[0] - 5)**2 + (theta[1] - 5)**2
    return jval

def grad(theta):
    gradient = np.zeros(2)
    gradient[0] = 2 * (theta[0]-5)
    gradient[1] = 2 * (theta[1]-5)
    return gradient 

init_theta = np.zeros(2)
out = fmin_bfgs(func,init_theta,grad,maxiter=10)

#print out
