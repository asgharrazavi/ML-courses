import numpy as np
import mdtraj.io as io
import scipy.optimize as op
import copy
import h5py


#================== softmax ====================================================
def fix_y(y,num_labels):							# y : (60000,10)
    yy = np.zeros((len(y),num_labels))
    for i in range(len(y)):
	yy[i,int(y[i])] = 1
    return yy.T

def softmaxCost(theta, numClasses, inputSize, lambdaa, inputData, labels):
    theta = np.reshape(theta, (numClasses, inputSize))
    m = len(labels)
    maxx = np.max(np.dot(theta,inputData))
    numer = np.exp(np.dot(theta,inputData) - maxx)
    denum = np.sum(np.exp(np.dot(theta,inputData) - maxx),axis=0).astype(float)
    yy = fix_y(labels,numClasses)
    cost = (-1.0/m) * np.sum(yy * np.log(numer/denum))
    cost += (lambdaa/2.0) * np.sum(theta**2)
    yy_p = yy - (numer/denum)
    thetagrad = np.dot(inputData,yy_p.T).T    
    thetagrad = (-1.0/m) * thetagrad
    thetagrad += lambdaa * theta
    return cost, thetagrad.flatten()
#===============================================================================

#-------------------------Training--------------------------------------------------------------
def train_nn():
    MaxIter = 100
    unrolled_theta = 0.005 * np.random.random((numClasses * inputSize, 1))
    out = op.fmin_l_bfgs_b(softmaxCost,unrolled_theta,fprime=None,args=(numClasses, inputSize, lambdaa, inputData, labels),maxfun=MaxIter, disp=1)
    theta = np.reshape(out[0], (numClasses, inputSize))
    np.savetxt('trained_theta.txt',theta)
    return theta
#train_nn()
#-----------------------------------------------------------------------------------------------
